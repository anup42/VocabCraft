"""Runtime loader for exact compact mT5 encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer, MT5EncoderModel

from vocabcraft.artifacts import read_json
from vocabcraft.exceptions import ArtifactError
from vocabcraft.fallback import FallbackPolicyName, ProfiledTokenizer
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer


class CompactMT5Encoder:
    """Reloaded compact encoder plus unchanged tokenizer and mapping guard."""

    def __init__(
        self,
        model: MT5EncoderModel,
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
    ) -> CompactMT5Encoder:
        """Load a completed encoder artifact and fail on mode mismatch."""

        root = Path(path)
        metadata = read_json(root / "vocabcraft-metadata.json")
        mapping_payload = read_json(root / "mapping-report.json")
        if not isinstance(metadata, dict) or metadata.get("vocabcraft_mode") != "encoder_exact":
            raise ArtifactError("artifact is not a VocabCraft encoder_exact profile")
        if not isinstance(mapping_payload, dict):
            raise ArtifactError("mapping report must be a JSON object")
        mapping = IdMapping.from_dict(mapping_payload)
        model = MT5EncoderModel.from_pretrained(root / "model")
        model.eval()
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

    def encode(self, text: str) -> tuple[Any | None, dict[str, Any]]:
        """Run compact inference only for a fully covered original token sequence."""

        decision = self.profiled_tokenizer.encode(text)
        if decision.compact_ids is None:
            return None, decision.to_dict()
        import torch

        input_ids = torch.tensor([decision.compact_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output, decision.to_dict()
