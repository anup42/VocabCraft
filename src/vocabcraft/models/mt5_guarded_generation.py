"""Greedy-only guarded mT5 generation with a full original output projection."""

from __future__ import annotations

import copy
from pathlib import Path
from threading import Lock, RLock
from typing import Any, cast
from weakref import WeakKeyDictionary

import torch
from transformers import GenerationConfig, MT5Config, MT5ForConditionalGeneration

from vocabcraft.artifacts import read_json
from vocabcraft.exceptions import ArtifactError, MissingTokenError, UnsupportedModeError
from vocabcraft.fallback import FallbackPolicyName, ProfiledTokenizer
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer

_MODEL_LOCKS: WeakKeyDictionary[torch.nn.Module, Any] = WeakKeyDictionary()
_MODEL_LOCKS_GUARD = Lock()


def load_guarded_mt5_model(
    model_path: str | Path, original_vocab_size: int
) -> MT5ForConditionalGeneration:
    """Load the nonstandard full-head/compact-embedding structure with exact dimensions."""

    root = Path(model_path)
    config_payload = read_json(root / "config.json")
    if not isinstance(config_payload, dict):
        raise ArtifactError("guarded model config must be a JSON object")
    if bool(config_payload.get("tie_word_embeddings", True)):
        raise ArtifactError("guarded model config unexpectedly requests tied output embeddings")
    config = MT5Config.from_dict(config_payload)
    config.tie_word_embeddings = False
    model = MT5ForConditionalGeneration(config)
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "in_features"):
        raise ArtifactError("guarded mT5 output head is not a supported linear projection")
    model.lm_head = torch.nn.Linear(
        int(output.in_features), original_vocab_size, bias=output.bias is not None
    )
    weights = root / "model.safetensors"
    if not weights.is_file():
        raise ArtifactError("guarded model requires a single model.safetensors file")
    from safetensors.torch import load_file

    model.load_state_dict(load_file(weights, device="cpu"), strict=True)
    # The output projection retains original IDs. The compact model config's
    # remapped token IDs must never drive output processors or stopping rules.
    generation_path = root.parent / "source-generation-config.json"
    if not generation_path.is_file():
        raise ArtifactError(
            "guarded model requires source-generation-config.json with original token IDs"
        )
    generation_payload = read_json(generation_path)
    if not isinstance(generation_payload, dict):
        raise ArtifactError("source generation config must be a JSON object")
    model.generation_config = GenerationConfig.from_dict(generation_payload)
    cast(Any, model)._vocabcraft_original_generation_config = True
    model.eval()
    return model


class GuardedMT5Generator:
    """Use Transformers greedy generation with original IDs and guarded embeddings.

    Calls through these wrappers are serialized for each compact model. Callers
    must not run the same compact model directly while a wrapper is generating.
    """

    def __init__(
        self,
        compact_model: MT5ForConditionalGeneration,
        tokenizer: OriginalTokenizer,
        mapping: IdMapping,
        profile_id: str,
        fallback_policy: FallbackPolicyName = "full_model",
        full_model: MT5ForConditionalGeneration | None = None,
        original_decoder_start_token_id: int = 0,
        original_eos_token_id: int = 1,
        *,
        original_generation_config: GenerationConfig | None = None,
    ) -> None:
        self.compact_model = compact_model.eval()
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.profiled_tokenizer = ProfiledTokenizer(tokenizer, mapping, profile_id, fallback_policy)
        self.fallback_policy = fallback_policy
        self.full_model = full_model.eval() if full_model is not None else None
        self.original_decoder_start_token_id = original_decoder_start_token_id
        self.original_eos_token_id = original_eos_token_id
        if (
            original_generation_config is None
            and full_model is None
            and mapping.new_to_old != tuple(range(mapping.original_vocab_size))
            and not getattr(compact_model, "_vocabcraft_original_generation_config", False)
        ):
            raise ArtifactError(
                "a nonidentity guarded model requires original_generation_config, "
                "a full model, or original generation settings loaded from its artifact"
            )
        self.original_generation_config = original_generation_config
        with _MODEL_LOCKS_GUARD:
            self._model_lock = _MODEL_LOCKS.setdefault(compact_model, RLock())

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        tokenizer: OriginalTokenizer,
        *,
        full_model: MT5ForConditionalGeneration | None = None,
        fallback_policy: FallbackPolicyName = "full_model",
    ) -> GuardedMT5Generator:
        """Load a guarded artifact with a caller-supplied unchanged tokenizer."""

        root = Path(path)
        metadata = read_json(root / "vocabcraft-metadata.json")
        mapping_payload = read_json(root / "mapping-report.json")
        if not isinstance(metadata, dict) or metadata.get("vocabcraft_mode") != "seq2seq_guarded":
            raise ArtifactError("artifact is not a VocabCraft seq2seq_guarded profile")
        if not isinstance(mapping_payload, dict):
            raise ArtifactError("mapping report must be a JSON object")
        mapping = IdMapping.from_dict(mapping_payload)
        model = load_guarded_mt5_model(root / "model", mapping.original_vocab_size)
        return cls(
            model,
            tokenizer,
            mapping,
            str(metadata.get("profile_id", "unknown")),
            fallback_policy,
            full_model,
            int(metadata["original_decoder_start_token_id"]),
            int(metadata["original_eos_token_id"]),
        )

    def _fallback(
        self,
        original_input_ids: list[int],
        reason: str,
        first_missing_generated_id: int | None,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        if self.fallback_policy == "full_model" and self.full_model is not None:
            full_input = torch.tensor(
                [original_input_ids], dtype=torch.long, device=self.full_model.device
            )
            with torch.no_grad():
                output_ids = self.full_model.generate(
                    input_ids=full_input,
                    attention_mask=torch.ones_like(full_input),
                    generation_config=copy.deepcopy(generation_config),
                )[0].tolist()
            return {
                "generated": True,
                "fallback_occurred": True,
                "fallback_policy": "full_model",
                "fallback_reason": reason,
                "first_missing_generated_id": first_missing_generated_id,
                "original_output_ids": output_ids,
                "text": self.tokenizer.decode(output_ids, skip_special_tokens=True),
            }
        return {
            "generated": False,
            "fallback_occurred": True,
            "fallback_policy": self.fallback_policy,
            "fallback_reason": reason,
            "first_missing_generated_id": first_missing_generated_id,
            "unsupported_profile": True,
        }

    def _generation_config(self, maximum_output_length: int) -> GenerationConfig:
        source = self.original_generation_config
        if source is None:
            source = (
                self.full_model.generation_config
                if self.full_model is not None
                else self.compact_model.generation_config
            )
        config = copy.deepcopy(source)
        if config.get_generation_mode() != "greedy_search":
            raise UnsupportedModeError(
                "seq2seq_guarded requires a greedy generation configuration; "
                "sampling, beam search and assisted generation are unsupported"
            )
        if config.num_return_sequences not in (None, 1):
            raise UnsupportedModeError("seq2seq_guarded requires num_return_sequences=1")
        for name in (
            "stop_strings",
            "token_healing",
            "custom_generate",
            "continuous_batching_config",
        ):
            if getattr(config, name, None):
                raise UnsupportedModeError(f"seq2seq_guarded does not support {name}")
        if getattr(config, "guidance_scale", None) not in (None, 1, 1.0):
            raise UnsupportedModeError("seq2seq_guarded does not support guidance_scale")
        # Wall-clock stopping is not invariant under compaction or a full restart.
        if config.max_time is not None:
            raise UnsupportedModeError("seq2seq_guarded does not support max_time")
        if config.cache_implementation == "paged":
            raise UnsupportedModeError("seq2seq_guarded does not support paged generation")
        config.do_sample = False
        config.num_beams = 1
        config.num_return_sequences = 1
        config.max_new_tokens = maximum_output_length
        config.return_dict_in_generate = False
        # Python embedding guards must execute on every decoder embedding call.
        config.disable_compile = True
        if config.decoder_start_token_id is None:
            config.decoder_start_token_id = config.bos_token_id
        if not isinstance(config.decoder_start_token_id, int):
            raise UnsupportedModeError("seq2seq_guarded requires one scalar decoder start ID")
        return config

    def _map_decoder_embedding_input(
        self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        del module
        original_ids = inputs[0]
        mapped = self.mapping.map_original_ids(original_ids.reshape(-1).tolist())
        return (
            torch.tensor(mapped, dtype=torch.long, device=original_ids.device).reshape_as(
                original_ids
            ),
            *inputs[1:],
        )

    def generate_greedy(
        self,
        text: str,
        maximum_output_length: int,
        *,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> dict[str, Any]:
        """Generate one request using original output IDs and a per-step embedding guard."""

        with self._model_lock:
            return self._generate_greedy(
                text, maximum_output_length, do_sample=do_sample, num_beams=num_beams
            )

    def _generate_greedy(
        self,
        text: str,
        maximum_output_length: int,
        *,
        do_sample: bool,
        num_beams: int,
    ) -> dict[str, Any]:
        if do_sample:
            raise UnsupportedModeError("seq2seq_guarded does not support sampling")
        if num_beams != 1:
            raise UnsupportedModeError("seq2seq_guarded does not support beam search")
        if maximum_output_length < 1:
            raise ValueError("maximum_output_length must be positive")
        generation_config = self._generation_config(maximum_output_length)
        decision = self.profiled_tokenizer.encode(text)
        if decision.compact_ids is None:
            return self._fallback(
                decision.original_ids,
                decision.fallback_reason or "input token is missing",
                None,
                generation_config,
            )
        try:
            self.mapping.map_original_ids([generation_config.decoder_start_token_id])
        except MissingTokenError as exc:
            raise ArtifactError("decoder start token is missing from guarded mapping") from exc
        device = self.compact_model.device
        compact_input = torch.tensor([decision.compact_ids], dtype=torch.long, device=device)
        original_input = torch.tensor([decision.original_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(compact_input)
        with torch.no_grad():
            # Run the encoder before installing the decoder hook: mT5 may share
            # the actual embedding module between its encoder and decoder.
            encoder_outputs = self.compact_model.get_encoder()(
                input_ids=compact_input,
                attention_mask=attention_mask,
                return_dict=True,
            )
            handle = self.compact_model.decoder.embed_tokens.register_forward_pre_hook(
                self._map_decoder_embedding_input
            )
            saved_generation_config = self.compact_model.generation_config
            self.compact_model.generation_config = generation_config
            missing: MissingTokenError | None = None
            try:
                output = self.compact_model.generate(
                    input_ids=original_input,
                    encoder_outputs=encoder_outputs,
                    generation_config=copy.deepcopy(generation_config),
                    attention_mask=attention_mask,
                )
                original_output_ids = output[0].tolist()
            except MissingTokenError as exc:
                missing = exc
            finally:
                handle.remove()
                self.compact_model.generation_config = saved_generation_config
        if missing is not None:
            return self._fallback(
                decision.original_ids,
                "generated original token has no compact decoder embedding",
                missing.missing_ids[0],
                generation_config,
            )
        return {
            "generated": True,
            "fallback_occurred": False,
            "fallback_policy": self.fallback_policy,
            "original_output_ids": original_output_ids,
            "text": self.tokenizer.decode(original_output_ids, skip_special_tokens=True),
            "greedy_only": True,
        }
