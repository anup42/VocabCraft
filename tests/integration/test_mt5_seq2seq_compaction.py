from pathlib import Path
from typing import Any

from transformers import MT5ForConditionalGeneration

from vocabcraft.evaluation.generation import compare_greedy_generation
from vocabcraft.evaluation.teacher_forcing import compare_teacher_forcing
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.selection import SelectionResult


def test_experimental_seq2seq_reload_logits_and_generation_reporting(
    mt5_adapter: MT5Adapter,
    en_hi_selection: tuple[SelectionResult, IdMapping],
    seq2seq_artifact: Path,
    first_sequence_record: Any,
) -> None:
    _, mapping = en_hi_selection
    compact = MT5ForConditionalGeneration.from_pretrained(seq2seq_artifact / "model").eval()
    source_ids = mt5_adapter.tokenizer.encode(first_sequence_record.source, add_special_tokens=True)
    target_ids = mt5_adapter.tokenizer.encode(first_sequence_record.target, add_special_tokens=True)
    teacher = compare_teacher_forcing(mt5_adapter.model, compact, source_ids, target_ids, mapping)
    assert teacher["source_covered"] and teacher["target_covered"]
    assert teacher["retained_raw_logits"]["maximum_absolute_difference"] <= 1.0e-5
    generation = compare_greedy_generation(
        mt5_adapter.model,
        compact,
        mt5_adapter.tokenizer,
        source_ids,
        mapping,
        maximum_output_length=8,
    )
    assert "first_divergence_position" in generation
    assert generation["experimental"]
