"""Numerical equivalence checks for covered compact encoder inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from vocabcraft.config import ValidationConfig
from vocabcraft.mappings import IdMapping


@dataclass(frozen=True)
class TensorComparison:
    """Numerical difference metrics for shape-identical tensors."""

    maximum_absolute_difference: float
    mean_absolute_difference: float
    maximum_relative_difference: float
    minimum_per_token_cosine_similarity: float
    sequence_cosine_similarity: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible representation."""

        return {
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "mean_absolute_difference": self.mean_absolute_difference,
            "maximum_relative_difference": self.maximum_relative_difference,
            "minimum_per_token_cosine_similarity": self.minimum_per_token_cosine_similarity,
            "sequence_cosine_similarity": self.sequence_cosine_similarity,
        }


def compare_tensors(original: torch.Tensor, compact: torch.Tensor) -> TensorComparison:
    """Compare tensors with absolute, relative, token, and sequence metrics."""

    if original.shape != compact.shape:
        raise ValueError(f"tensor shapes differ: {tuple(original.shape)} != {tuple(compact.shape)}")
    left = original.detach().to(dtype=torch.float64)
    right = compact.detach().to(dtype=torch.float64)
    difference = (left - right).abs()
    denominator = torch.maximum(left.abs(), torch.full_like(left, torch.finfo(torch.float64).eps))
    relative = difference / denominator
    if left.ndim >= 2:
        token_cosine = functional.cosine_similarity(left, right, dim=-1, eps=1.0e-12)
        minimum_token_cosine = float(token_cosine.min().item())
    else:
        minimum_token_cosine = float(
            functional.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1.0e-12).item()
        )
    sequence_cosine = float(
        functional.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1.0e-12).item()
    )
    return TensorComparison(
        maximum_absolute_difference=float(difference.max().item()) if difference.numel() else 0.0,
        mean_absolute_difference=float(difference.mean().item()) if difference.numel() else 0.0,
        maximum_relative_difference=float(relative.max().item()) if relative.numel() else 0.0,
        minimum_per_token_cosine_similarity=minimum_token_cosine,
        sequence_cosine_similarity=sequence_cosine,
    )


def _encoder(model: Any) -> Any:
    getter = getattr(model, "get_encoder", None)
    return getter() if callable(getter) else model


def compare_encoder_models(
    original_model: Any,
    compact_model: Any,
    original_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mapping: IdMapping,
    tolerances: ValidationConfig,
    *,
    include_hidden_states: bool = False,
) -> dict[str, Any]:
    """Compare unchanged original IDs against their compact mapped IDs in eval mode."""

    original_encoder = _encoder(original_model)
    compact_encoder = _encoder(compact_model)
    original_model.eval()
    compact_model.eval()
    compact_rows = [
        mapping.map_original_ids([int(token_id) for token_id in row.tolist()])
        for row in original_input_ids
    ]
    compact_input_ids = torch.tensor(
        compact_rows, dtype=torch.long, device=original_input_ids.device
    )
    with torch.no_grad():
        original_embedding = original_encoder.embed_tokens(original_input_ids)
        compact_embedding = compact_encoder.embed_tokens(compact_input_ids)
        original_output = original_encoder(
            input_ids=original_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=include_hidden_states,
            return_dict=True,
        )
        compact_output = compact_encoder(
            input_ids=compact_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=include_hidden_states,
            return_dict=True,
        )
    embedding_metrics = compare_tensors(original_embedding, compact_embedding)
    final_metrics = compare_tensors(
        original_output.last_hidden_state, compact_output.last_hidden_state
    )
    hidden_metrics: list[dict[str, float]] = []
    if include_hidden_states:
        original_hidden = original_output.hidden_states or ()
        compact_hidden = compact_output.hidden_states or ()
        if len(original_hidden) != len(compact_hidden):
            raise ValueError("encoder hidden-state counts differ")
        hidden_metrics = [
            compare_tensors(left, right).to_dict()
            for left, right in zip(original_hidden, compact_hidden, strict=True)
        ]
    passed = (
        final_metrics.maximum_absolute_difference <= tolerances.encoder_max_absolute_difference
        and final_metrics.mean_absolute_difference <= tolerances.encoder_mean_absolute_difference
        and final_metrics.sequence_cosine_similarity >= tolerances.minimum_encoder_cosine_similarity
    )
    return {
        "passed": passed,
        "final_hidden_state": final_metrics.to_dict(),
        "embedding_layer": embedding_metrics.to_dict(),
        "intermediate_hidden_states": hidden_metrics,
        "attention_mask_preserved": True,
        "tolerances": {
            "maximum_absolute_difference": tolerances.encoder_max_absolute_difference,
            "mean_absolute_difference": tolerances.encoder_mean_absolute_difference,
            "minimum_cosine_similarity": tolerances.minimum_encoder_cosine_similarity,
        },
    }
