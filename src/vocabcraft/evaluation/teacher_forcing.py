"""Teacher-forced compact seq2seq comparison for covered source/target IDs."""

from __future__ import annotations

from typing import Any

import torch

from vocabcraft.evaluation.encoder_equivalence import compare_tensors
from vocabcraft.mappings import IdMapping


def compare_teacher_forcing(
    original_model: Any,
    compact_model: Any,
    source_original_ids: list[int],
    target_original_ids: list[int],
    mapping: IdMapping,
) -> dict[str, Any]:
    """Compare retained raw logits and losses while flagging softmax limitations."""

    source_compact_ids = mapping.map_original_ids(source_original_ids)
    target_compact_ids = mapping.map_labels(target_original_ids)
    original_model.eval()
    compact_model.eval()
    original_input = torch.tensor([source_original_ids], dtype=torch.long)
    compact_input = torch.tensor([source_compact_ids], dtype=torch.long)
    original_labels = torch.tensor([target_original_ids], dtype=torch.long)
    compact_labels = torch.tensor([target_compact_ids], dtype=torch.long)
    retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
    with torch.no_grad():
        original_output = original_model(input_ids=original_input, labels=original_labels)
        compact_output = compact_model(input_ids=compact_input, labels=compact_labels)
    retained_original_logits = original_output.logits.index_select(-1, retained)
    logits = compare_tensors(retained_original_logits, compact_output.logits)
    return {
        "retained_raw_logits": logits.to_dict(),
        "original_loss": float(original_output.loss.item()),
        "compact_loss": float(compact_output.loss.item()),
        "loss_difference": float(compact_output.loss.item() - original_output.loss.item()),
        "source_covered": True,
        "target_covered": True,
        "warning": (
            "Retained raw-logit similarity does not prove equal softmax probabilities after "
            "output-vocabulary reduction."
        ),
    }
