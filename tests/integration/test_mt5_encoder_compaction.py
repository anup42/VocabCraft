from pathlib import Path

import torch
from transformers import MT5EncoderModel

from vocabcraft.config import VocabCraftConfig
from vocabcraft.evaluation.datasets import stream_jsonl
from vocabcraft.evaluation.encoder_equivalence import compare_encoder_models
from vocabcraft.fallback import guard_original_ids
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.selection import SelectionResult


def test_encoder_exact_rows_equivalence_and_explicit_fallback(
    mt5_adapter: MT5Adapter,
    en_hi_config: VocabCraftConfig,
    en_hi_selection: tuple[SelectionResult, IdMapping],
    encoder_artifact: Path,
) -> None:
    _, mapping = en_hi_selection
    compact = MT5EncoderModel.from_pretrained(encoder_artifact / "model").eval()
    retained = torch.tensor(mapping.new_to_old, dtype=torch.long)
    assert torch.equal(
        compact.get_input_embeddings().weight,
        mt5_adapter.model.shared.weight.detach().index_select(0, retained),
    )
    first = next(iter(stream_jsonl(en_hi_config.profile.calibration_files[0])))
    source_ids = mt5_adapter.tokenizer.encode(first.source, add_special_tokens=True)
    original_ids = torch.tensor([source_ids], dtype=torch.long)
    comparison = compare_encoder_models(
        mt5_adapter.model,
        compact,
        original_ids,
        torch.ones_like(original_ids),
        mapping,
        en_hi_config.validation,
        include_hidden_states=True,
    )
    assert comparison["passed"]
    unsupported_ids = mt5_adapter.tokenizer.encode("東京へ行きます", add_special_tokens=True)
    decision = guard_original_ids(
        unsupported_ids,
        mapping,
        mt5_adapter.tokenizer,
        en_hi_config.profile.id,
        "full_model",
    )
    assert decision.fallback_required
    assert decision.compact_ids is None
