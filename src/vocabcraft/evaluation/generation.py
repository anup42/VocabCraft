"""Deterministic generation comparison retaining every divergence."""

from __future__ import annotations

from typing import Any

import torch

from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def compare_greedy_generation(
    original_model: Any,
    compact_model: Any,
    tokenizer: OriginalTokenizer,
    source_original_ids: list[int],
    mapping: IdMapping,
    *,
    maximum_output_length: int,
) -> dict[str, Any]:
    """Run greedy generation on both models and report, never discard, divergence."""

    if maximum_output_length < 1:
        raise ValueError("maximum_output_length must be positive")
    source_compact_ids = mapping.map_original_ids(source_original_ids)
    original_input = torch.tensor([source_original_ids], dtype=torch.long)
    compact_input = torch.tensor([source_compact_ids], dtype=torch.long)
    original_model.eval()
    compact_model.eval()
    with torch.no_grad():
        original_ids = original_model.generate(
            input_ids=original_input,
            do_sample=False,
            num_beams=1,
            max_new_tokens=maximum_output_length,
        )[0].tolist()
        compact_ids = compact_model.generate(
            input_ids=compact_input,
            do_sample=False,
            num_beams=1,
            max_new_tokens=maximum_output_length,
        )[0].tolist()
    mapped_compact_ids = mapping.map_compact_ids(compact_ids)
    original_text = tokenizer.decode(original_ids, skip_special_tokens=True)
    compact_text = tokenizer.decode(mapped_compact_ids, skip_special_tokens=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return {
        "exact_original_token_id_match": original_ids == mapped_compact_ids,
        "exact_decoded_text_match": original_text == compact_text,
        "first_divergence_position": _first_divergence(original_ids, mapped_compact_ids),
        "original_output_length": len(original_ids),
        "compact_output_length": len(mapped_compact_ids),
        "eos_difference": (eos_id in original_ids) != (eos_id in mapped_compact_ids),
        "original_ids": original_ids,
        "compact_ids": compact_ids,
        "mapped_compact_original_ids": mapped_compact_ids,
        "original_text": original_text,
        "compact_text": compact_text,
        "experimental": True,
    }
