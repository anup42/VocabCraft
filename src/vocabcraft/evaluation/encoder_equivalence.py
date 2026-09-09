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
    if original.numel() == 0:
        raise ValueError("cannot establish equivalence from empty tensors")
    left = original.detach().to(dtype=torch.float64)
    right = compact.detach().to(dtype=torch.float64)
    difference = (left - right).abs()
    denominator = torch.maximum(left.abs(), torch.full_like(left, torch.finfo(torch.float64).eps))
    relative = difference / denominator
    if left.ndim >= 2:
        token_cosine = functional.cosine_similarity(left, right, dim=-1, eps=1.0e-12)
        both_zero = (left.norm(dim=-1) == 0) & (right.norm(dim=-1) == 0)
        token_cosine = torch.where(both_zero, torch.ones_like(token_cosine), token_cosine)
        minimum_token_cosine = float(token_cosine.min().item())
    else:
        minimum_token_cosine = float(
            functional.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1.0e-12).item()
        )
    sequence_cosine = float(
        functional.cosine_similarity(left.flatten(), right.flatten(), dim=0, eps=1.0e-12).item()
    )
    if left.norm() == 0 and right.norm() == 0:
        sequence_cosine = 1.0
        if left.ndim < 2:
            minimum_token_cosine = 1.0
    return TensorComparison(
        maximum_absolute_difference=float(difference.max().item()) if difference.numel() else 0.0,
        mean_absolute_difference=float(difference.mean().item()) if difference.numel() else 0.0,
        maximum_relative_difference=float(relative.max().item()) if relative.numel() else 0.0,
        minimum_per_token_cosine_similarity=minimum_token_cosine,
        sequence_cosine_similarity=sequence_cosine,
    )


def _encoder(model: Any) -> Any:
    getter = getattr(model, "get_encoder", None)
    # Encoder-only XLM-R also exposes get_encoder(), but it returns only the
    # transformer blocks, without the input embedding/position pipeline.
    return (
        getter()
        if (
            getattr(model.config, "is_encoder_decoder", False)
            or getattr(model.config, "model_type", None) == "mt5"
        )
        and callable(getter)
        else model
    )


def _passes(metrics: TensorComparison, tolerances: ValidationConfig) -> bool:
    return (
        metrics.maximum_absolute_difference <= tolerances.encoder_max_absolute_difference
        and metrics.mean_absolute_difference <= tolerances.encoder_mean_absolute_difference
        and metrics.sequence_cosine_similarity >= tolerances.minimum_encoder_cosine_similarity
        and metrics.minimum_per_token_cosine_similarity
        >= tolerances.minimum_encoder_cosine_similarity
    )


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0e-9)


def compare_encoder_models(
    original_model: Any,
    compact_model: Any,
    original_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mapping: IdMapping,
    tolerances: ValidationConfig,
    *,
    include_hidden_states: bool = False,
    include_sentence_embeddings: bool = False,
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
        original_embedding = original_encoder.get_input_embeddings()(original_input_ids)
        compact_embedding = compact_encoder.get_input_embeddings()(compact_input_ids)
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
    visible = attention_mask.bool()
    embedding_metrics = compare_tensors(original_embedding[visible], compact_embedding[visible])
    final_metrics = compare_tensors(
        original_output.last_hidden_state[visible], compact_output.last_hidden_state[visible]
    )
    hidden_metrics: list[dict[str, float]] = []
    if include_hidden_states:
        original_hidden = original_output.hidden_states or ()
        compact_hidden = compact_output.hidden_states or ()
        if len(original_hidden) != len(compact_hidden):
            raise ValueError("encoder hidden-state counts differ")
        hidden_metrics = [
            compare_tensors(left[visible], right[visible]).to_dict()
            for left, right in zip(original_hidden, compact_hidden, strict=True)
        ]
    per_example = []
    for index in range(original_input_ids.shape[0]):
        selected = attention_mask[index].bool()
        metrics = compare_tensors(
            original_output.last_hidden_state[index, selected],
            compact_output.last_hidden_state[index, selected],
        )
        item: dict[str, Any] = {
            "passed": _passes(metrics, tolerances),
            "final_hidden_state": metrics.to_dict(),
            "embedding_layer": compare_tensors(
                original_embedding[index, selected], compact_embedding[index, selected]
            ).to_dict(),
            "attention_mask_preserved": True,
            "intermediate_hidden_states": [
                compare_tensors(left[index, selected], right[index, selected]).to_dict()
                for left, right in zip(
                    original_output.hidden_states or (),
                    compact_output.hidden_states or (),
                    strict=True,
                )
            ],
        }
        if include_sentence_embeddings:
            left = _mean_pool(
                original_output.last_hidden_state[index : index + 1],
                attention_mask[index : index + 1],
            )
            right = _mean_pool(
                compact_output.last_hidden_state[index : index + 1],
                attention_mask[index : index + 1],
            )
            pooled = compare_tensors(left, right)
            normalized = compare_tensors(
                functional.normalize(left, dim=-1), functional.normalize(right, dim=-1)
            )
            item["sentence_embedding"] = pooled.to_dict()
            item["normalized_sentence_embedding"] = normalized.to_dict()
            item["passed"] = (
                item["passed"] and _passes(pooled, tolerances) and _passes(normalized, tolerances)
            )
        per_example.append(item)
    return {
        "passed": _passes(final_metrics, tolerances)
        and all(item["passed"] for item in per_example),
        "final_hidden_state": final_metrics.to_dict(),
        "embedding_layer": embedding_metrics.to_dict(),
        "intermediate_hidden_states": hidden_metrics,
        "attention_mask_preserved": True,
        "per_example": per_example,
        "tolerances": {
            "maximum_absolute_difference": tolerances.encoder_max_absolute_difference,
            "mean_absolute_difference": tolerances.encoder_mean_absolute_difference,
            "minimum_cosine_similarity": tolerances.minimum_encoder_cosine_similarity,
        },
    }
