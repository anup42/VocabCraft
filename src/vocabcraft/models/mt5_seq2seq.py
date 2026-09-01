"""Runtime loader for experimental compact mT5 sequence-to-sequence artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer, MT5Config, MT5ForConditionalGeneration

from vocabcraft.artifacts import read_json
from vocabcraft.exceptions import ArtifactError
from vocabcraft.fallback import FallbackPolicyName, ProfiledTokenizer
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer


def load_compact_mt5_seq2seq_model(path: str | Path) -> MT5ForConditionalGeneration:
    """Reload local mT5 while preserving the serialized output-tying decision."""

    root = Path(path)
    config_payload = read_json(root / "config.json")
    if not isinstance(config_payload, dict):
        raise ArtifactError("compact mT5 config must be a JSON object")
    config = MT5Config.from_dict(config_payload)
    config.tie_word_embeddings = bool(config_payload.get("tie_word_embeddings", True))
    model = MT5ForConditionalGeneration.from_pretrained(root, config=config)
    model.eval()
    return model


class CompactMT5Seq2Seq:
    """Experimental compact model with original-ID guarding and decoding."""

    def __init__(
        self,
        model: MT5ForConditionalGeneration,
        tokenizer: Any,
        mapping: IdMapping,
        profiled_tokenizer: ProfiledTokenizer,
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.profiled_tokenizer = profiled_tokenizer
        self.metadata = metadata

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        fallback_policy: FallbackPolicyName = "full_model",
    ) -> CompactMT5Seq2Seq:
        """Load an experimental compact seq2seq artifact."""

        root = Path(path)
        metadata = read_json(root / "vocabcraft-metadata.json")
        mapping_payload = read_json(root / "mapping-report.json")
        if not isinstance(metadata, dict) or metadata.get("vocabcraft_mode") != "seq2seq_compact":
            raise ArtifactError("artifact is not a VocabCraft seq2seq_compact profile")
        if not isinstance(mapping_payload, dict):
            raise ArtifactError("mapping report must be a JSON object")
        mapping = IdMapping.from_dict(mapping_payload)
        model = load_compact_mt5_seq2seq_model(root / "model")
        tokenizer = cast(
            OriginalTokenizer,
            AutoTokenizer.from_pretrained(root / "tokenizer", use_fast=False),
        )
        profiled = ProfiledTokenizer(
            tokenizer,
            mapping,
            str(metadata.get("profile_id", "unknown")),
            fallback_policy,
        )
        return cls(model, tokenizer, mapping, profiled, metadata)

    def generate_greedy(self, text: str, maximum_output_length: int) -> dict[str, Any]:
        """Generate deterministically; this remains empirical, not equivalent generation."""

        if maximum_output_length < 1:
            raise ValueError("maximum_output_length must be positive")
        decision = self.profiled_tokenizer.encode(text)
        if decision.compact_ids is None:
            return {"guard": decision.to_dict(), "generated": False}
        import torch

        input_ids = torch.tensor([decision.compact_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            compact_output_ids = (
                cast(Any, self.model)
                .generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=maximum_output_length,
                )[0]
                .tolist()
            )
        original_output_ids = self.mapping.map_compact_ids(compact_output_ids)
        return {
            "guard": decision.to_dict(),
            "generated": True,
            "compact_output_ids": compact_output_ids,
            "original_output_ids": original_output_ids,
            "text": self.tokenizer.decode(original_output_ids, skip_special_tokens=True),
            "experimental": True,
        }
