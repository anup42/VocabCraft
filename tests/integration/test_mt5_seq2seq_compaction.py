from pathlib import Path
from typing import Any

from vocabcraft.config import VocabCraftConfig
from vocabcraft.evaluation.generation import compare_greedy_generation
from vocabcraft.evaluation.teacher_forcing import compare_teacher_forcing
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.models.mt5_seq2seq import load_compact_mt5_seq2seq_model
from vocabcraft.selection import SelectionResult


def test_experimental_seq2seq_reload_logits_and_generation_reporting(
    mt5_adapter: MT5Adapter,
    en_hi_config: VocabCraftConfig,
    en_hi_selection: tuple[SelectionResult, IdMapping],
    seq2seq_artifact: Path,
    first_sequence_record: Any,
) -> None:
    _, mapping = en_hi_selection
    compact = load_compact_mt5_seq2seq_model(seq2seq_artifact / "model")
    source_ids = mt5_adapter.tokenizer.encode(first_sequence_record.source, add_special_tokens=True)
    target_ids = mt5_adapter.tokenizer.encode(first_sequence_record.target, add_special_tokens=True)
    teacher = compare_teacher_forcing(
        mt5_adapter.model,
        compact,
        source_ids,
        target_ids,
        mapping,
        en_hi_config.validation.teacher_forcing_max_absolute_difference,
    )
    assert teacher["source_covered"] and teacher["target_covered"]
    assert teacher["passed"]
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
