"""Runtime-inspected mT5 model adapter."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
from collections import defaultdict
from pathlib import Path
from typing import Any

import sentencepiece  # type: ignore[import-untyped]
import torch
import transformers
from transformers import AutoTokenizer, MT5EncoderModel, MT5ForConditionalGeneration

from vocabcraft.artifacts import atomic_output_directory, write_json
from vocabcraft.exceptions import (
    InspectionError,
    UnsupportedModeError,
    UnsupportedModelError,
    ValidationFailure,
)
from vocabcraft.hashing import sha256_bytes, sha256_json
from vocabcraft.mappings import IdMapping, remap_token_id_fields
from vocabcraft.models.base import ModelAdapter
from vocabcraft.selection import SelectionResult

_GENERATION_ID_FIELDS = (
    "pad_token_id",
    "eos_token_id",
    "bos_token_id",
    "decoder_start_token_id",
    "unk_token_id",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "suppress_tokens",
    "begin_suppress_tokens",
    "bad_words_ids",
)


def _shape(value: Any | None) -> list[int] | None:
    weight = getattr(value, "weight", None)
    return list(weight.shape) if isinstance(weight, torch.Tensor) else None


def _storage_pointer(tensor: torch.Tensor | None) -> int | None:
    if tensor is None:
        return None
    return tensor.untyped_storage().data_ptr()


def _tensor_weight(value: Any | None) -> torch.Tensor | None:
    weight = getattr(value, "weight", None)
    return weight if isinstance(weight, torch.Tensor) else None


def _hash_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor metadata and bytes in bounded row chunks."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        value = tensor.detach().cpu().contiguous()
        if value.ndim == 0:
            digest.update(value.view(torch.uint8).numpy().tobytes())
            continue
        row_bytes = max(1, value[0].numel() * value.element_size()) if value.shape[0] else 1
        rows_per_chunk = max(1, (16 * 1024 * 1024) // row_bytes)
        for start in range(0, value.shape[0], rows_per_chunk):
            chunk = value[start : start + rows_per_chunk].contiguous().view(torch.uint8)
            digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _tokenizer_hash(tokenizer: Any) -> str:
    processor = getattr(tokenizer, "sp_model", None)
    serialized = getattr(processor, "serialized_model_proto", None)
    if callable(serialized):
        data = serialized()
        if isinstance(data, bytes):
            return sha256_bytes(data)
    pieces = [tokenizer.convert_ids_to_tokens(token_id) for token_id in range(len(tokenizer))]
    return sha256_json(pieces)


def _sentencepiece_details(tokenizer: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": importlib.metadata.version("sentencepiece"),
        "model_type": None,
        "normalization": None,
    }
    processor = getattr(tokenizer, "sp_model", None)
    serialized = getattr(processor, "serialized_model_proto", None)
    if not callable(serialized):
        return result
    try:
        from sentencepiece import sentencepiece_model_pb2

        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(serialized())
        result["model_type"] = sentencepiece_model_pb2.TrainerSpec.ModelType.Name(
            proto.trainer_spec.model_type
        )
        result["normalization"] = {
            "name": proto.normalizer_spec.name,
            "add_dummy_prefix": proto.normalizer_spec.add_dummy_prefix,
            "remove_extra_whitespaces": proto.normalizer_spec.remove_extra_whitespaces,
            "escape_whitespaces": proto.normalizer_spec.escape_whitespaces,
        }
    except (AttributeError, ImportError, TypeError, ValueError):
        result["normalization"] = "unavailable from this tokenizer implementation"
    return result


def _generation_ids(generation_config: Any | None) -> dict[str, Any]:
    if generation_config is None:
        return {}
    return {
        name: value
        for name in _GENERATION_ID_FIELDS
        if (value := getattr(generation_config, name, None)) is not None
    }


def _special_ids(tokenizer: Any, config: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "all_special_ids": list(tokenizer.all_special_ids),
        "all_special_tokens": list(tokenizer.all_special_tokens),
        "additional_special_tokens": list(tokenizer.additional_special_tokens),
    }
    for name in (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
        "decoder_start_token_id",
    ):
        result[name] = getattr(config, name, getattr(tokenizer, name, None))
    result["extra_ids"] = sorted(
        token_id
        for token_id in tokenizer.all_special_ids
        if str(tokenizer.convert_ids_to_tokens(token_id)).startswith("<extra_id_")
    )
    return result


class MT5Adapter(ModelAdapter):
    """Adapter that validates actual mT5 objects before compaction."""

    def __init__(self, model: MT5ForConditionalGeneration, tokenizer: Any, identifier: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.identifier = identifier
        self._model_state_hash: str | None = None
        self._tokenizer_hash: str | None = None
        self.model.eval()
        self.validate_structure()

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> MT5Adapter:
        """Load a supported mT5 checkpoint on CPU without remote code execution."""

        load_arguments: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "revision": revision,
        }
        tokenizer = AutoTokenizer.from_pretrained(identifier, use_fast=False, **load_arguments)
        model = MT5ForConditionalGeneration.from_pretrained(
            identifier,
            device_map=None,
            **load_arguments,
        )
        model.cpu()
        return cls(model, tokenizer, identifier)

    @property
    def model_vocab_size(self) -> int:
        """Return the inspected configuration vocabulary size."""

        value = getattr(self.model.config, "vocab_size", None)
        if not isinstance(value, int) or value <= 0:
            raise InspectionError("mT5 config.vocab_size must be a positive integer")
        return value

    def validate_structure(self) -> None:
        """Reject checkpoints that differ from supported mT5 structures."""

        if not isinstance(self.model, MT5ForConditionalGeneration):
            raise UnsupportedModelError(
                f"expected MT5ForConditionalGeneration, loaded {type(self.model).__name__}"
            )
        shared = getattr(self.model, "shared", None)
        encoder = getattr(getattr(self.model, "encoder", None), "embed_tokens", None)
        decoder = getattr(getattr(self.model, "decoder", None), "embed_tokens", None)
        output = self.model.get_output_embeddings()
        for value, name in (
            (shared, "shared embedding"),
            (encoder, "encoder embedding"),
            (decoder, "decoder embedding"),
            (output, "output embedding/lm_head"),
        ):
            weight = _tensor_weight(value)
            if weight is None or weight.ndim != 2:
                raise UnsupportedModelError(f"mT5 {name} must expose a rank-2 weight")
        if _tensor_weight(shared).shape[0] != self.model_vocab_size:  # type: ignore[union-attr]
            raise UnsupportedModelError("shared embedding rows do not equal config.vocab_size")
        for entry in self.list_vocabulary_tensors():
            axes = entry["vocabulary_axes"]
            if any(axis != 0 for axis in axes):
                raise UnsupportedModelError(
                    f"vocabulary tensor {entry['name']} uses unsupported non-leading axis {axes}"
                )

    def list_vocabulary_tensors(self) -> list[dict[str, Any]]:
        """List every state tensor with any model-vocabulary-sized dimension."""

        vocabulary_size = self.model_vocab_size
        result: list[dict[str, Any]] = []
        for name, tensor in self.model.state_dict().items():
            axes = [axis for axis, size in enumerate(tensor.shape) if size == vocabulary_size]
            if axes:
                result.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "vocabulary_axes": axes,
                        "is_bias": tensor.ndim == 1,
                        "bytes": tensor.numel() * tensor.element_size(),
                    }
                )
        return result

    def detect_weight_tying(self) -> dict[str, Any]:
        """Inspect identity and storage pointers for all mT5 vocabulary modules."""

        shared = self.model.shared
        encoder = self.model.encoder.embed_tokens
        decoder = self.model.decoder.embed_tokens
        output = self.model.get_output_embeddings()
        shared_weight = _tensor_weight(shared)
        output_weight = _tensor_weight(output)
        return {
            "shared_is_encoder_embedding": shared is encoder,
            "shared_is_decoder_embedding": shared is decoder,
            "encoder_decoder_same_object": encoder is decoder,
            "shared_encoder_same_storage": _storage_pointer(shared_weight)
            == _storage_pointer(_tensor_weight(encoder)),
            "shared_decoder_same_storage": _storage_pointer(shared_weight)
            == _storage_pointer(_tensor_weight(decoder)),
            "shared_output_same_storage": _storage_pointer(shared_weight)
            == _storage_pointer(output_weight),
            "config_tie_word_embeddings": bool(
                getattr(self.model.config, "tie_word_embeddings", False)
            ),
            "pointers": {
                "shared": _storage_pointer(shared_weight),
                "encoder": _storage_pointer(_tensor_weight(encoder)),
                "decoder": _storage_pointer(_tensor_weight(decoder)),
                "output": _storage_pointer(output_weight),
            },
        }

    def inspect_model(self) -> dict[str, Any]:
        """Return a source- and runtime-grounded mT5 inspection report."""

        state = self.model.state_dict()
        vocabulary_tensors = self.list_vocabulary_tensors()
        unique_vocab_parameters: dict[int, torch.Tensor] = {}
        for name, parameter in self.model.named_parameters():
            del name
            if any(size == self.model_vocab_size for size in parameter.shape):
                unique_vocab_parameters[parameter.untyped_storage().data_ptr()] = parameter
        total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        vocabulary_parameters = sum(
            parameter.numel() for parameter in unique_vocab_parameters.values()
        )
        dtype_bytes: defaultdict[str, int] = defaultdict(int)
        for tensor in state.values():
            dtype_bytes[str(tensor.dtype)] += tensor.numel() * tensor.element_size()
        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        sentencepiece_details = _sentencepiece_details(self.tokenizer)
        return {
            "model_identifier": self.identifier,
            "resolved_revision": resolved_revision,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "sentencepiece_version": sentencepiece.__version__,
            "sentencepiece_model_type": sentencepiece_details["model_type"],
            "sentencepiece_normalization": sentencepiece_details["normalization"],
            "tokenizer_implementation": type(self.tokenizer).__module__,
            "tokenizer_vocabulary_size": len(self.tokenizer),
            "configuration_vocabulary_size": self.model_vocab_size,
            "embedding_dimension": self.model.shared.weight.shape[1],
            "shared_embedding_shape": _shape(self.model.shared),
            "encoder_embedding_shape": _shape(self.model.encoder.embed_tokens),
            "decoder_embedding_shape": _shape(self.model.decoder.embed_tokens),
            "output_head_shape": _shape(self.model.get_output_embeddings()),
            "weight_tying": self.detect_weight_tying(),
            "vocabulary_sized_tensors": vocabulary_tensors,
            "vocabulary_sized_biases": [item for item in vocabulary_tensors if item["is_bias"]],
            "special_token_ids": _special_ids(self.tokenizer, self.model.config),
            "generation_configuration_token_ids": _generation_ids(
                getattr(self.model, "generation_config", None)
            ),
            "model_dtype": str(next(self.model.parameters()).dtype),
            "total_parameter_count": total_parameters,
            "vocabulary_dependent_parameter_count": vocabulary_parameters,
            "non_vocabulary_parameter_count": total_parameters - vocabulary_parameters,
            "estimated_state_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
            "model_state_sha256": self.model_state_hash(),
            "tokenizer_sha256": self.tokenizer_hash(),
            "tokenizer_model_vocab_size_mismatch": len(self.tokenizer) != self.model_vocab_size,
        }

    def model_state_hash(self) -> str:
        """Return and cache the exact source state hash for this loaded adapter."""

        if self._model_state_hash is None:
            self._model_state_hash = _hash_state_dict(self.model.state_dict())
        return self._model_state_hash

    def tokenizer_hash(self) -> str:
        """Return and cache the exact serialized tokenizer/model hash."""

        if self._tokenizer_hash is None:
            self._tokenizer_hash = _tokenizer_hash(self.tokenizer)
        return self._tokenizer_hash

    def _compact_state_dict(
        self, target_state: dict[str, torch.Tensor], mapping: IdMapping
    ) -> dict[str, torch.Tensor]:
        source_state = self.model.state_dict()
        retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
        result: dict[str, torch.Tensor] = {}
        for name, target in target_state.items():
            source = source_state.get(name)
            if source is None:
                raise UnsupportedModelError(f"target state contains unknown tensor {name}")
            if source.shape == target.shape:
                result[name] = source.detach().clone()
            elif (
                source.ndim >= 1
                and source.shape[0] == mapping.original_vocab_size
                and target.shape[0] == len(mapping.new_to_old)
                and source.shape[1:] == target.shape[1:]
            ):
                result[name] = source.detach().index_select(0, retained).clone()
            else:
                raise UnsupportedModelError(
                    f"cannot safely compact tensor {name}: source {tuple(source.shape)}, "
                    f"target {tuple(target.shape)}"
                )
        return result

    @staticmethod
    def _apply_remapped_configuration(owner: Any, remapped: dict[str, Any]) -> None:
        for name, value in remapped.items():
            setattr(owner, name, value)

    def _compact_config(self, mapping: IdMapping) -> Any:
        config = copy.deepcopy(self.model.config)
        self._apply_remapped_configuration(config, remap_token_id_fields(config, mapping))
        config.vocab_size = len(mapping.new_to_old)
        return config

    def _write_common_artifacts(
        self, staging: Path, mapping: IdMapping, mode: str, profile_id: str
    ) -> dict[str, Any]:
        tokenizer_dir = staging / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_dir)
        write_json(staging / "source-config.json", self.model.config.to_dict())
        if getattr(self.model, "generation_config", None) is not None:
            write_json(
                staging / "source-generation-config.json",
                self.model.generation_config.to_dict(),
            )
        mapping_payload = mapping.to_dict()
        write_json(staging / "mapping-report.json", mapping_payload)
        metadata = {
            "format_version": 1,
            "product": "VocabCraft",
            "vocabcraft_mode": mode,
            "profile_id": profile_id,
            "source_model": self.identifier,
            "source_revision": getattr(self.model.config, "_commit_hash", None),
            "model_state_sha256": self.model_state_hash(),
            "tokenizer_sha256": self.tokenizer_hash(),
            "original_model_vocabulary_size": mapping.original_vocab_size,
            "original_tokenizer_vocabulary_size": len(self.tokenizer),
            "compact_vocabulary_size": len(mapping.new_to_old),
            "original_sentencepiece_preserved": True,
        }
        write_json(staging / "vocabcraft-metadata.json", metadata)
        return metadata

    def build_encoder_profile(
        self,
        mapping: IdMapping,
        output: Path,
        profile_id: str = "unspecified",
        selection: SelectionResult | None = None,
    ) -> dict[str, Any]:
        """Build an encoder-only artifact with exact copied retained rows."""

        if mapping.original_vocab_size != self.model_vocab_size:
            raise ValidationFailure("mapping original size does not match model vocabulary size")
        config = self._compact_config(mapping)
        compact = MT5EncoderModel(config)
        compact_state = self._compact_state_dict(compact.state_dict(), mapping)
        compact.load_state_dict(compact_state, strict=True)
        compact.eval()
        with atomic_output_directory(output) as staging:
            model_dir = staging / "model"
            compact.save_pretrained(model_dir, safe_serialization=True)
            metadata = self._write_common_artifacts(staging, mapping, "encoder_exact", profile_id)
            if selection is not None:
                from vocabcraft.manifests import write_selection_artifacts

                write_selection_artifacts(staging, selection, profile_id)
            reloaded = MT5EncoderModel.from_pretrained(model_dir).eval()
            retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
            expected = self.model.shared.weight.detach().index_select(0, retained)
            if not torch.equal(reloaded.get_input_embeddings().weight, expected):
                raise ValidationFailure("reloaded compact encoder rows differ from source rows")
            metadata["reloaded_row_copy_exact"] = True
            metadata["serialized_model_bytes"] = sum(
                path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
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
        """Build an experimental fully compact mT5 encoder-decoder artifact."""

        if mapping.original_vocab_size != self.model_vocab_size:
            raise ValidationFailure("mapping original size does not match model vocabulary size")
        config = self._compact_config(mapping)
        compact = MT5ForConditionalGeneration(config)
        generation_config = copy.deepcopy(self.model.generation_config)
        self._apply_remapped_configuration(
            generation_config, remap_token_id_fields(generation_config, mapping)
        )
        compact.generation_config = generation_config
        compact_state = self._compact_state_dict(compact.state_dict(), mapping)
        compact.load_state_dict(compact_state, strict=True)
        compact.eval()
        with atomic_output_directory(output) as staging:
            model_dir = staging / "model"
            compact.save_pretrained(model_dir, safe_serialization=True)
            metadata = self._write_common_artifacts(staging, mapping, "seq2seq_compact", profile_id)
            if selection is not None:
                from vocabcraft.manifests import write_selection_artifacts

                write_selection_artifacts(staging, selection, profile_id)
            reloaded = MT5ForConditionalGeneration.from_pretrained(model_dir).eval()
            retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
            expected = self.model.shared.weight.detach().index_select(0, retained)
            if not torch.equal(reloaded.shared.weight, expected):
                raise ValidationFailure("reloaded compact shared rows differ from source rows")
            metadata.update(
                {
                    "experimental": True,
                    "reloaded_row_copy_exact": True,
                    "serialized_model_bytes": sum(
                        path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
                    ),
                    "warning": (
                        "Compact output vocabularies can alter greedy, beam, "
                        "and sampled generation."
                    ),
                }
            )
            write_json(staging / "vocabcraft-metadata.json", metadata)
        return metadata

    def build_guarded_profile(
        self,
        mapping: IdMapping,
        output: Path,
        profile_id: str = "unspecified",
        selection: SelectionResult | None = None,
    ) -> dict[str, Any]:
        """Build a greedy-only guarded artifact when the output head is truly untied."""

        tying = self.detect_weight_tying()
        if tying["shared_output_same_storage"] or tying["config_tie_word_embeddings"]:
            raise UnsupportedModeError(
                "seq2seq_guarded is unsafe for this checkpoint because the full output "
                "projection is tied to the compact shared embedding"
            )
        if mapping.original_vocab_size != self.model_vocab_size:
            raise ValidationFailure("mapping original size does not match model vocabulary size")
        source_output = self.model.get_output_embeddings()
        if source_output is None or not isinstance(source_output, torch.nn.Linear):
            raise UnsupportedModeError("guarded mode requires an untied linear output head")
        config = self._compact_config(mapping)
        config.tie_word_embeddings = False
        compact = MT5ForConditionalGeneration(config)
        compact.lm_head = torch.nn.Linear(
            source_output.in_features,
            self.model_vocab_size,
            bias=source_output.bias is not None,
        )
        compact_state = self._compact_state_dict(compact.state_dict(), mapping)
        compact.load_state_dict(compact_state, strict=True)
        compact.eval()
        with atomic_output_directory(output) as staging:
            model_dir = staging / "model"
            compact.save_pretrained(model_dir, safe_serialization=True)
            metadata = self._write_common_artifacts(staging, mapping, "seq2seq_guarded", profile_id)
            if selection is not None:
                from vocabcraft.manifests import write_selection_artifacts

                write_selection_artifacts(staging, selection, profile_id)
            decoder_start = getattr(self.model.config, "decoder_start_token_id", None)
            eos = getattr(self.model.config, "eos_token_id", None)
            if not isinstance(decoder_start, int) or not isinstance(eos, int):
                raise UnsupportedModeError(
                    "guarded mode requires scalar decoder_start_token_id and eos_token_id"
                )
            metadata.update(
                {
                    "experimental": True,
                    "guarded_scope": {
                        "do_sample": False,
                        "num_beams": 1,
                        "maximum_output_length_required": True,
                    },
                    "original_decoder_start_token_id": decoder_start,
                    "original_eos_token_id": eos,
                    "full_original_output_projection": True,
                }
            )
            from vocabcraft.models.mt5_guarded_generation import load_guarded_mt5_model

            reloaded = load_guarded_mt5_model(model_dir, self.model_vocab_size)
            retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
            expected = self.model.shared.weight.detach().index_select(0, retained)
            if not torch.equal(reloaded.shared.weight, expected):
                raise ValidationFailure("reloaded guarded shared rows differ from source rows")
            if not torch.equal(reloaded.lm_head.weight, source_output.weight):
                raise ValidationFailure("reloaded guarded full output head differs from source")
            metadata["reloaded_row_copy_exact"] = True
            write_json(staging / "vocabcraft-metadata.json", metadata)
        return metadata
