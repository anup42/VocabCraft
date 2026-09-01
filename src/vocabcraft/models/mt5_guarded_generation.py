"""Greedy-only guarded mT5 generation with a full original output projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from vocabcraft.artifacts import read_json
from vocabcraft.exceptions import ArtifactError, MissingTokenError, UnsupportedModeError
from vocabcraft.fallback import FallbackPolicyName, ProfiledTokenizer
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer


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
    model.eval()
    return model


class GuardedMT5Generator:
    """Select original output IDs, guard decoder embeddings, and rerun on full model."""

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
    ) -> None:
        self.compact_model = compact_model.eval()
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.profiled_tokenizer = ProfiledTokenizer(tokenizer, mapping, profile_id, fallback_policy)
        self.fallback_policy = fallback_policy
        self.full_model = full_model.eval() if full_model is not None else None
        self.original_decoder_start_token_id = original_decoder_start_token_id
        self.original_eos_token_id = original_eos_token_id

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
        maximum_output_length: int,
    ) -> dict[str, Any]:
        if self.fallback_policy == "full_model" and self.full_model is not None:
            full_input = torch.tensor([original_input_ids], dtype=torch.long)
            with torch.no_grad():
                output_ids = self.full_model.generate(
                    input_ids=full_input,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=maximum_output_length,
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

    def generate_greedy(
        self,
        text: str,
        maximum_output_length: int,
        *,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> dict[str, Any]:
        """Generate one request using original output IDs and a per-step embedding guard."""

        if do_sample:
            raise UnsupportedModeError("seq2seq_guarded does not support sampling")
        if num_beams != 1:
            raise UnsupportedModeError("seq2seq_guarded does not support beam search")
        if maximum_output_length < 1:
            raise ValueError("maximum_output_length must be positive")
        decision = self.profiled_tokenizer.encode(text)
        if decision.compact_ids is None:
            return self._fallback(
                decision.original_ids,
                decision.fallback_reason or "input token is missing",
                None,
                maximum_output_length,
            )
        try:
            decoder_compact_ids = self.mapping.map_original_ids(
                [self.original_decoder_start_token_id]
            )
        except MissingTokenError as exc:
            raise ArtifactError("decoder start token is missing from guarded mapping") from exc
        original_output_ids = [self.original_decoder_start_token_id]
        compact_input = torch.tensor([decision.compact_ids], dtype=torch.long)
        attention_mask = torch.ones_like(compact_input)
        for _ in range(maximum_output_length):
            decoder_input = torch.tensor([decoder_compact_ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.compact_model(
                    input_ids=compact_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input,
                    use_cache=False,
                    return_dict=True,
                ).logits
            next_original_id = int(torch.argmax(logits[0, -1]).item())
            original_output_ids.append(next_original_id)
            if next_original_id == self.original_eos_token_id:
                break
            compact_id = self.mapping.old_to_new.get(next_original_id)
            if compact_id is None:
                return self._fallback(
                    decision.original_ids,
                    "generated original token has no compact decoder embedding",
                    next_original_id,
                    maximum_output_length,
                )
            decoder_compact_ids.append(compact_id)
        return {
            "generated": True,
            "fallback_occurred": False,
            "fallback_policy": self.fallback_policy,
            "original_output_ids": original_output_ids,
            "text": self.tokenizer.decode(original_output_ids, skip_special_tokens=True),
            "greedy_only": True,
        }
