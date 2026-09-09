"""Exact word-embedding row compaction for XLM-R and mean-pooled sentence encoders."""

from __future__ import annotations

import copy
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer, XLMRobertaModel

from vocabcraft.artifacts import atomic_output_directory, read_json, write_json
from vocabcraft.exceptions import (
    ArtifactError,
    UnsupportedModeError,
    UnsupportedModelError,
    ValidationFailure,
)
from vocabcraft.fallback import FallbackPolicyName, ProfiledTokenizer
from vocabcraft.hashing import sha256_json
from vocabcraft.mappings import IdMapping, remap_token_id_fields
from vocabcraft.models.base import ModelAdapter
from vocabcraft.models.mt5 import _hash_state_dict, _sentencepiece_details, _special_ids
from vocabcraft.selection import SelectionResult
from vocabcraft.tokenizers.base import OriginalTokenizer
from vocabcraft.tokenizers.fingerprint import tokenizer_behavior_hash

WORD_EMBEDDING_NAME = "embeddings.word_embeddings.weight"
SENTENCE_CONFIG_NAME = "sentence-embedding-config.json"
_DTYPE_ARGUMENT = "dtype" if int(transformers.__version__.split(".", 1)[0]) >= 5 else "torch_dtype"


def load_xlm_roberta_model(path: str | Path, *, add_pooling_layer: bool) -> XLMRobertaModel:
    """Load every saved tensor strictly, preserving whether a dense pooler exists."""

    dtype_args: dict[str, Any] = {_DTYPE_ARGUMENT: "auto"}
    model, info = cast(
        tuple[XLMRobertaModel, dict[str, Any]],
        XLMRobertaModel.from_pretrained(
            path,
            add_pooling_layer=add_pooling_layer,
            output_loading_info=True,
            **dtype_args,
        ),
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(field):
            raise ArtifactError(f"XLM-R checkpoint has {field}: {info[field]}")
    return cast(XLMRobertaModel, model.eval())


def pool_sentence_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    normalize_embeddings: bool = False,
) -> torch.Tensor:
    """Apply Sentence Transformers mean pooling, optionally followed by L2 normalization."""

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    embeddings = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    if normalize_embeddings:
        embeddings = functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def _tokenize_embedding_batch(
    tokenizer: Any,
    texts: list[str],
    config: dict[str, Any] | None,
    prompt: str | None = None,
    prompt_name: str | None = None,
) -> dict[str, Any]:
    if config is None:
        if prompt is not None or prompt_name is not None:
            raise UnsupportedModeError("this artifact has no sentence prompt configuration")
        return dict(tokenizer(texts, padding=True, return_attention_mask=True))
    if prompt is not None and prompt_name is not None:
        raise ValueError("specify prompt or prompt_name, not both")
    if prompt is None:
        name = prompt_name if prompt_name is not None else config["default_prompt_name"]
        if name is not None and name not in config["prompts"]:
            raise ValueError(f"unknown sentence prompt: {name!r}")
        prompt = config["prompts"][name] if name is not None else ""
    prepared = [(prompt + text).strip() for text in texts]
    if config["do_lower_case"]:
        prepared = [text.lower() for text in prepared]
    return dict(
        tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=config["max_seq_length"],
            return_attention_mask=True,
        )
    )


def tokenize_embedding_text(
    tokenizer: Any, text: str, sentence_embedding_config: dict[str, Any] | None
) -> list[int]:
    """Tokenize one source sentence with its saved prompt and truncation behavior."""

    encoded = _tokenize_embedding_batch(tokenizer, [text], sentence_embedding_config)
    return cast(list[int], encoded["input_ids"][0])


def _source_pooler_presence(path: Path) -> bool | None:
    """Read local safe checkpoint headers without materializing training state."""

    single = path / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as checkpoint:
            names = checkpoint.keys()
            return any(name.startswith("pooler.") for name in names)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        payload = read_json(index)
        if isinstance(payload, dict) and isinstance(payload.get("weight_map"), dict):
            return any(name.startswith("pooler.") for name in payload["weight_map"])
    return None


def _sentence_configuration(
    path: Path, tokenizer: Any, model: XLMRobertaModel
) -> dict[str, Any] | None:
    modules_file = path / "modules.json"
    if not modules_file.is_file():
        return None
    modules = read_json(modules_file)
    if not isinstance(modules, list) or len(modules) != 2:
        raise UnsupportedModelError(
            "sentence encoder requires exactly Transformer and mean Pooling"
        )
    for index, expected in enumerate(("Transformer", "Pooling")):
        item = modules[index]
        if (
            not isinstance(item, dict)
            or item.get("idx") != index
            or not str(item.get("type", "")).startswith("sentence_transformers.")
            or str(item.get("type", "")).split(".")[-1] != expected
        ):
            raise UnsupportedModelError(f"unsupported sentence encoder module at position {index}")
    if modules[0].get("path", "") not in ("", "."):
        raise UnsupportedModelError("sentence Transformer must be stored in the model root")
    pool_path = (path / str(modules[1].get("path", ""))).resolve()
    if not pool_path.is_relative_to(path.resolve()):
        raise UnsupportedModelError("pooling module path escapes the source model directory")
    pooling = read_json(pool_path / "config.json")
    if not isinstance(pooling, dict):
        raise UnsupportedModelError("pooling configuration must be an object")
    legacy_flags = {key: value for key, value in pooling.items() if key.startswith("pooling_mode_")}
    if pooling.get("pooling_mode", "mean") != "mean" or (
        legacy_flags
        and (
            legacy_flags.get("pooling_mode_mean_tokens") is not True
            or any(
                value for key, value in legacy_flags.items() if key != "pooling_mode_mean_tokens"
            )
        )
    ):
        raise UnsupportedModelError("only attention-masked mean sentence pooling is supported")
    dimension = pooling.get("embedding_dimension", pooling.get("word_embedding_dimension"))
    if dimension != model.config.hidden_size:
        raise UnsupportedModelError("sentence pooling dimension differs from the encoder")
    if pooling.get("include_prompt", True) is not True:
        raise UnsupportedModelError("sentence pooling with include_prompt=false is unsupported")
    transformer_file = path / "sentence_bert_config.json"
    transformer = read_json(transformer_file) if transformer_file.is_file() else {}
    sentence_file = path / "config_sentence_transformers.json"
    sentence = read_json(sentence_file) if sentence_file.is_file() else {}
    if not isinstance(transformer, dict) or not isinstance(sentence, dict):
        raise UnsupportedModelError("sentence encoder configurations must be objects")
    if sentence.get("truncate_dim") not in (None, model.config.hidden_size):
        raise UnsupportedModelError("sentence embedding dimension truncation is unsupported")
    if type(transformer.get("do_lower_case", False)) is not bool:
        raise UnsupportedModelError("sentence do_lower_case must be a boolean")
    if transformer.get("transformer_task", "feature-extraction") != "feature-extraction":
        raise UnsupportedModelError("only feature-extraction sentence Transformers are supported")
    if transformer.get("module_output_name", "token_embeddings") != "token_embeddings":
        raise UnsupportedModelError("unsupported sentence Transformer output name")
    modalities = transformer.get("modality_config")
    if modalities is not None and modalities != {
        "text": {"method": "forward", "method_output_name": "last_hidden_state"}
    }:
        raise UnsupportedModelError("unsupported sentence Transformer modality configuration")
    max_positions = model.config.max_position_embeddings - model.config.pad_token_id - 1
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", max_positions))
    max_length = transformer.get("max_seq_length", min(max_positions, tokenizer_limit))
    if not isinstance(max_length, int) or not 2 <= max_length <= max_positions:
        raise UnsupportedModelError("sentence max_seq_length exceeds XLM-R position capacity")
    prompts = sentence.get("prompts", {})
    default_prompt = sentence.get("default_prompt_name")
    if not isinstance(prompts, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in prompts.items()
    ):
        raise UnsupportedModelError("sentence prompts must map names to strings")
    if default_prompt is not None and (
        not isinstance(default_prompt, str) or default_prompt not in prompts
    ):
        raise UnsupportedModelError("default sentence prompt is absent from saved prompts")
    return {
        "pooling_mode": "mean",
        "embedding_dimension": dimension,
        "include_prompt": True,
        "prompts": prompts,
        "default_prompt_name": default_prompt,
        "max_seq_length": max_length,
        "do_lower_case": bool(transformer.get("do_lower_case", False)),
        "source_pooling_config": pooling,
        "source_transformer_config": transformer,
        "source_sentence_transformers_config": sentence,
    }


class XLMRobertaAdapter(ModelAdapter):
    """Inspect and compact only XLM-R word embeddings; retain every other tensor."""

    def __init__(self, model: XLMRobertaModel, tokenizer: Any, identifier: str) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.identifier = identifier
        self._model_state_hash: str | None = None
        self._tokenizer_hash: str | None = None
        self.validate_structure()
        self.sentence_embedding_config = _sentence_configuration(Path(identifier), tokenizer, model)

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> XLMRobertaAdapter:
        """Load a local or Hub XLM-R encoder; reject missing or incompatible weights."""

        args: dict[str, Any] = {"revision": revision, "trust_remote_code": trust_remote_code}
        tokenizer = AutoTokenizer.from_pretrained(identifier, use_fast=True, **args)
        args[_DTYPE_ARGUMENT] = "auto"
        pooler = _source_pooler_presence(Path(identifier))
        model, info = cast(
            tuple[XLMRobertaModel, dict[str, Any]],
            XLMRobertaModel.from_pretrained(
                identifier,
                add_pooling_layer=pooler is not False,
                output_loading_info=True,
                **args,
            ),
        )
        missing = info.get("missing_keys", [])
        if pooler is None and set(missing) == {"pooler.dense.weight", "pooler.dense.bias"}:
            model.pooler = None
            missing = []
        if missing or any(
            info.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise UnsupportedModelError(f"XLM-R source weights do not match the encoder: {info}")
        return cls(model.cpu(), tokenizer, identifier)

    @property
    def model_vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def validate_structure(self) -> None:
        if not isinstance(self.model, XLMRobertaModel) or self.model.config.is_decoder:
            raise UnsupportedModelError("expected an encoder-only XLMRobertaModel")
        embeddings = self.model.embeddings
        if embeddings.word_embeddings.weight.shape != (
            self.model_vocab_size,
            self.model.config.hidden_size,
        ):
            raise UnsupportedModelError("XLM-R word embeddings do not match configured dimensions")
        pad = self.model.config.pad_token_id
        if not isinstance(pad, int) or not 0 <= pad < self.model_vocab_size:
            raise UnsupportedModelError("XLM-R requires a valid padding token ID")
        if any(
            value != pad
            for value in (
                embeddings.padding_idx,
                embeddings.word_embeddings.padding_idx,
                embeddings.position_embeddings.padding_idx,
                self.tokenizer.pad_token_id,
            )
        ):
            raise UnsupportedModelError("XLM-R model/tokenizer/position padding indices disagree")
        if getattr(self.model.config, "position_embedding_type", "absolute") != "absolute":
            raise UnsupportedModelError("only absolute XLM-R position embeddings are supported")
        if len(self.tokenizer) > self.model_vocab_size:
            raise UnsupportedModelError("tokenizer IDs exceed the XLM-R word embedding table")

    def list_vocabulary_tensors(self) -> list[dict[str, Any]]:
        """Use the actual word embedding, never coincidentally equal hidden/position dimensions."""

        weight = self.model.embeddings.word_embeddings.weight
        return [
            {
                "name": WORD_EMBEDDING_NAME,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "vocabulary_axes": [0],
                "is_bias": False,
                "bytes": weight.numel() * weight.element_size(),
            }
        ]

    def detect_weight_tying(self) -> dict[str, Any]:
        return {
            "encoder_input_only": True,
            "shared_encoder_same_storage": True,
            "shared_output_same_storage": False,
            "config_tie_word_embeddings": bool(self.model.config.tie_word_embeddings),
            "has_output_head": False,
            "has_pooler": self.model.pooler is not None,
        }

    def model_state_hash(self) -> str:
        if self._model_state_hash is None:
            self._model_state_hash = _hash_state_dict(self.model.state_dict())
        return self._model_state_hash

    def tokenizer_hash(self) -> str:
        if self._tokenizer_hash is None:
            self._tokenizer_hash = tokenizer_behavior_hash(self.tokenizer)
        return self._tokenizer_hash

    def tokenize_text(self, text: str) -> list[int]:
        """Apply the same saved sentence preprocessing used by compact inference."""

        return tokenize_embedding_text(self.tokenizer, text, self.sentence_embedding_config)

    def inspect_model(self) -> dict[str, Any]:
        word = self.model.embeddings.word_embeddings.weight
        total = sum(parameter.numel() for parameter in self.model.parameters())
        dtype_bytes: defaultdict[str, int] = defaultdict(int)
        for value in self.model.state_dict().values():
            dtype_bytes[str(value.dtype)] += value.numel() * value.element_size()
        sentencepiece = _sentencepiece_details(self.tokenizer)
        return {
            "model_identifier": self.identifier,
            "model_family": "xlm-roberta",
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "sentencepiece_version": sentencepiece["version"],
            "sentencepiece_model_type": sentencepiece["model_type"],
            "sentencepiece_normalization": sentencepiece["normalization"],
            "tokenizer_implementation": type(self.tokenizer).__module__,
            "tokenizer_vocabulary_size": len(self.tokenizer),
            "configuration_vocabulary_size": self.model_vocab_size,
            "embedding_dimension": word.shape[1],
            "shared_embedding_shape": list(word.shape),
            "encoder_embedding_shape": list(word.shape),
            "decoder_embedding_shape": None,
            "output_head_shape": None,
            "weight_tying": self.detect_weight_tying(),
            "vocabulary_sized_tensors": self.list_vocabulary_tensors(),
            "vocabulary_sized_biases": [],
            "special_token_ids": _special_ids(self.tokenizer, self.model.config),
            "generation_configuration_token_ids": {},
            "model_dtype": str(word.dtype),
            "total_parameter_count": total,
            "vocabulary_dependent_parameter_count": word.numel(),
            "non_vocabulary_parameter_count": total - word.numel(),
            "estimated_state_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
            "model_state_sha256": self.model_state_hash(),
            "tokenizer_sha256": self.tokenizer_hash(),
            "tokenizer_behavior_sha256": self.tokenizer_hash(),
            "tokenizer_model_vocab_size_mismatch": len(self.tokenizer) != self.model_vocab_size,
            "position_padding_idx": self.model.embeddings.padding_idx,
            "sentence_embedding_config": self.sentence_embedding_config,
        }

    def build_encoder_profile(
        self,
        mapping: IdMapping,
        output: Path,
        profile_id: str = "unspecified",
        selection: SelectionResult | None = None,
    ) -> dict[str, Any]:
        """Copy retained word rows and verify every tensor after native model reload."""

        mapping.validate()
        if mapping.original_vocab_size != self.model_vocab_size:
            raise ValidationFailure("mapping original size does not match model vocabulary size")
        mandatory = set(self.tokenizer.all_special_ids)
        missing = mandatory.difference(mapping.old_to_new)
        if missing:
            raise ValidationFailure(f"XLM-R mapping must retain all special IDs: {sorted(missing)}")
        pad = self.model.config.pad_token_id
        if mapping.old_to_new.get(pad) != pad:
            raise UnsupportedModeError(
                "XLM-R compaction must preserve the padding ID to keep position indices exact"
            )
        config = copy.deepcopy(self.model.config)
        for key, value in remap_token_id_fields(config, mapping).items():
            setattr(config, key, value)
        config.vocab_size = len(mapping.new_to_old)
        pooler = self.model.pooler is not None
        compact = XLMRobertaModel(config, add_pooling_layer=pooler)
        cast(Any, compact).to(dtype=self.model.embeddings.word_embeddings.weight.dtype)
        retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
        compact_state = {
            name: tensor.detach().index_select(0, retained).clone()
            if name == WORD_EMBEDDING_NAME
            else tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }
        compact.load_state_dict(compact_state, strict=True)
        del compact_state
        gc.collect()
        with atomic_output_directory(output) as staging:
            model_dir = staging / "model"
            compact.eval().save_pretrained(model_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(staging / "tokenizer")
            write_json(staging / "mapping-report.json", mapping.to_dict())
            write_json(staging / "source-config.json", self.model.config.to_dict())
            metadata = {
                "format_version": 1,
                "product": "VocabCraft",
                "vocabcraft_mode": "encoder_exact",
                "model_family": "xlm-roberta",
                "profile_id": profile_id,
                "source_model": self.identifier,
                "source_revision": getattr(self.model.config, "_commit_hash", None),
                "model_state_sha256": self.model_state_hash(),
                "tokenizer_sha256": self.tokenizer_hash(),
                "tokenizer_behavior_sha256": self.tokenizer_hash(),
                "original_model_vocabulary_size": mapping.original_vocab_size,
                "original_tokenizer_vocabulary_size": len(self.tokenizer),
                "compact_vocabulary_size": len(mapping.new_to_old),
                "original_tokenizer_preserved": True,
                "original_sentencepiece_preserved": True,
                "add_pooling_layer": pooler,
                "position_padding_idx": pad,
                "sentence_embedding_config": self.sentence_embedding_config,
                "sentence_embedding_sha256": sha256_json(self.sentence_embedding_config),
            }
            write_json(staging / SENTENCE_CONFIG_NAME, self.sentence_embedding_config)
            if selection is not None:
                from vocabcraft.manifests import write_selection_artifacts

                write_selection_artifacts(staging, selection, profile_id)
            del compact
            gc.collect()
            reloaded = load_xlm_roberta_model(model_dir, add_pooling_layer=pooler)
            source_state = self.model.state_dict()
            reloaded_state = reloaded.state_dict()
            if set(source_state) != set(reloaded_state):
                raise ValidationFailure("reloaded XLM-R has different state tensor names")
            for name, source in source_state.items():
                expected = (
                    source.index_select(0, retained) if name == WORD_EMBEDDING_NAME else source
                )
                if reloaded_state[name].dtype != expected.dtype or not torch.equal(
                    reloaded_state[name], expected
                ):
                    raise ValidationFailure(f"reloaded XLM-R tensor differs from source: {name}")
            saved_tokenizer = AutoTokenizer.from_pretrained(staging / "tokenizer", use_fast=True)
            if tokenizer_behavior_hash(saved_tokenizer) != self.tokenizer_hash():
                raise ValidationFailure("serialized XLM-R tokenizer behavior changed")
            metadata.update(
                {
                    "reloaded_row_copy_exact": True,
                    "reloaded_all_tensors_exact": True,
                    "serialized_model_bytes": sum(
                        item.stat().st_size for item in model_dir.rglob("*") if item.is_file()
                    ),
                }
            )
            write_json(staging / "vocabcraft-metadata.json", metadata)
        return metadata

    def build_seq2seq_profile(
        self,
        mapping: IdMapping,
        output: Path,
        profile_id: str = "unspecified",
        selection: SelectionResult | None = None,
    ) -> dict[str, Any]:
        raise UnsupportedModeError("XLM-R is encoder-only and does not support seq2seq_compact")

    def build_guarded_profile(
        self,
        mapping: IdMapping,
        output: Path,
        profile_id: str = "unspecified",
        selection: SelectionResult | None = None,
    ) -> dict[str, Any]:
        raise UnsupportedModeError("XLM-R is encoder-only and does not support seq2seq_guarded")


class CompactXLMRobertaEncoder:
    """Unchanged tokenizer, coverage guard, compact XLM-R, and saved sentence mean pooling."""

    def __init__(
        self,
        model: XLMRobertaModel,
        tokenizer: Any,
        mapping: IdMapping,
        profiled_tokenizer: ProfiledTokenizer,
        metadata: dict[str, Any],
        sentence_embedding_config: dict[str, Any] | None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.profiled_tokenizer = profiled_tokenizer
        self.metadata = metadata
        self.sentence_embedding_config = sentence_embedding_config

    @classmethod
    def from_artifact(
        cls, path: str | Path, fallback_policy: FallbackPolicyName = "full_model"
    ) -> CompactXLMRobertaEncoder:
        root = Path(path)
        metadata = read_json(root / "vocabcraft-metadata.json")
        if not isinstance(metadata, dict) or (
            metadata.get("vocabcraft_mode") != "encoder_exact"
            or metadata.get("model_family") != "xlm-roberta"
        ):
            raise ArtifactError("artifact is not a VocabCraft XLM-R encoder_exact profile")
        payload = read_json(root / "mapping-report.json")
        if not isinstance(payload, dict):
            raise ArtifactError("mapping report must be a JSON object")
        mapping = IdMapping.from_dict(payload)
        model = load_xlm_roberta_model(
            root / "model", add_pooling_layer=bool(metadata["add_pooling_layer"])
        )
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", use_fast=True)
        if tokenizer_behavior_hash(tokenizer) != metadata.get("tokenizer_behavior_sha256"):
            raise ArtifactError("saved XLM-R tokenizer behavior does not match the source")
        if model.config.vocab_size != len(mapping.new_to_old) or (
            metadata.get("original_model_vocabulary_size") != mapping.original_vocab_size
        ):
            raise ArtifactError("XLM-R artifact mapping and model dimensions disagree")
        pad = metadata.get("position_padding_idx")
        if (
            model.embeddings.padding_idx != pad
            or tokenizer.pad_token_id != pad
            or mapping.old_to_new.get(pad) != pad
        ):
            raise ArtifactError("XLM-R artifact changes the original position padding index")
        if set(tokenizer.all_special_ids).difference(mapping.old_to_new):
            raise ArtifactError("XLM-R artifact mapping omits required special tokens")
        sentence = read_json(root / SENTENCE_CONFIG_NAME)
        if sha256_json(sentence) != metadata.get("sentence_embedding_sha256"):
            raise ArtifactError("saved sentence embedding configuration checksum differs")
        if sentence is not None and not isinstance(sentence, dict):
            raise ArtifactError("sentence embedding configuration must be an object or null")
        profiled = ProfiledTokenizer(
            cast(OriginalTokenizer, tokenizer),
            mapping,
            str(metadata.get("profile_id", "unknown")),
            fallback_policy,
        )
        return cls(model, tokenizer, mapping, profiled, metadata, sentence)

    def _tokenize(
        self, texts: list[str], prompt: str | None, prompt_name: str | None
    ) -> dict[str, Any]:
        return _tokenize_embedding_batch(
            self.tokenizer, texts, self.sentence_embedding_config, prompt, prompt_name
        )

    def encode(
        self, text: str, *, prompt: str | None = None, prompt_name: str | None = None
    ) -> tuple[Any | None, dict[str, Any]]:
        """Return token-level outputs only after guarding all tokens consumed by the encoder."""

        encoded = self._tokenize([text], prompt, prompt_name)
        decision = self.profiled_tokenizer.guard_batch(encoded["input_ids"])[0]
        if decision.compact_ids is None:
            return None, decision.to_dict()
        device = next(self.model.parameters()).device
        features = {
            key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in encoded.items()
            if key in {"attention_mask", "token_type_ids"}
        }
        features["input_ids"] = torch.tensor([decision.compact_ids], device=device)
        with torch.no_grad():
            output = self.model(**features)
        return output, decision.to_dict()

    def embed_batch(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = False,
        prompt: str | None = None,
        prompt_name: str | None = None,
    ) -> tuple[list[torch.Tensor | None], list[dict[str, Any]]]:
        """Return embeddings and fallback decisions in input order, preserving padding masks."""

        if self.sentence_embedding_config is None:
            raise UnsupportedModeError("sentence embedding requires a saved mean-pooling module")
        if not texts:
            return [], []
        encoded = self._tokenize(texts, prompt, prompt_name)
        decisions = self.profiled_tokenizer.guard_batch(encoded["input_ids"])
        supported = [index for index, item in enumerate(decisions) if item.compact_ids is not None]
        results: list[torch.Tensor | None] = [None] * len(texts)
        if supported:
            device = next(self.model.parameters()).device
            features = {
                key: torch.tensor([value[index] for index in supported], device=device)
                for key, value in encoded.items()
                if key in {"attention_mask", "token_type_ids"}
            }
            features["input_ids"] = torch.tensor(
                [decisions[index].compact_ids for index in supported], device=device
            )
            with torch.no_grad():
                output = self.model(**features)
                embeddings = pool_sentence_embeddings(
                    output.last_hidden_state, features["attention_mask"], normalize_embeddings
                )
            for offset, index in enumerate(supported):
                results[index] = embeddings[offset]
        return results, [item.to_dict() for item in decisions]

    def embed(
        self,
        text: str,
        *,
        normalize_embeddings: bool = False,
        prompt: str | None = None,
        prompt_name: str | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Return one sentence vector or an explicit fallback decision."""

        embeddings, decisions = self.embed_batch(
            [text],
            normalize_embeddings=normalize_embeddings,
            prompt=prompt,
            prompt_name=prompt_name,
        )
        return embeddings[0], decisions[0]
