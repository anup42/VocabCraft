from pathlib import Path

import torch
from test_mt5_models import _tiny_model
from transformers import MT5EncoderModel, MT5ForConditionalGeneration

from vocabcraft.config import ValidationConfig
from vocabcraft.evaluation.coverage import evaluate_coverage
from vocabcraft.evaluation.encoder_equivalence import compare_encoder_models, compare_tensors
from vocabcraft.evaluation.external_hook import redact_secrets
from vocabcraft.evaluation.non_inferiority import paired_bootstrap_non_inferiority
from vocabcraft.evaluation.teacher_forcing import compare_teacher_forcing
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter


def test_coverage_reports_fallback_without_new_unk(fake_tokenizer: object, tmp_path: Path) -> None:
    data = tmp_path / "data.jsonl"
    data.write_text(
        '{"id":"covered","text":"hello Ravi","language":"en","domain":"chat"}\n'
        '{"id":"missing","text":"東京","language":"ja","domain":"fallback",'
        '"critical":true}\n',
        encoding="utf-8",
    )
    mapping = IdMapping.from_retained([0, 1, 2, 3, 8], 11)
    report = evaluate_coverage(  # type: ignore[arg-type]
        fake_tokenizer,
        mapping,
        [data],
        profile_id="test",
        fallback_policy="full_model",
    )
    payload = report.to_dict()
    assert payload["fully_covered_example_percentage"] == 50.0
    assert report.missing_critical_examples == ["missing"]
    assert report.missing_ids_by_count == {6: 1}
    assert report.new_unk_count == 0
    assert report.fallback_examples == 1


def test_tensor_comparison_reports_exact_equality() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    result = compare_tensors(value, value.clone())
    assert result.maximum_absolute_difference == 0.0
    assert result.mean_absolute_difference == 0.0
    assert result.minimum_per_token_cosine_similarity > 0.999999


def test_encoder_equivalence_for_copied_rows(fake_tokenizer: object, tmp_path: Path) -> None:
    original = _tiny_model()
    adapter = MT5Adapter(original, fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 11)
    artifact = tmp_path / "encoder"
    adapter.build_encoder_profile(mapping, artifact, "test")
    compact = MT5EncoderModel.from_pretrained(artifact / "model")
    original_ids = torch.tensor([[3, 8, 1]], dtype=torch.long)
    attention_mask = torch.ones_like(original_ids)
    result = compare_encoder_models(
        original,
        compact,
        original_ids,
        attention_mask,
        mapping,
        ValidationConfig(),
        include_hidden_states=True,
    )
    assert result["passed"]
    assert result["final_hidden_state"]["maximum_absolute_difference"] == 0.0
    assert result["embedding_layer"]["maximum_absolute_difference"] == 0.0


def test_teacher_forcing_compares_retained_logits(fake_tokenizer: object, tmp_path: Path) -> None:
    original = _tiny_model()
    adapter = MT5Adapter(original, fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 11)
    artifact = tmp_path / "seq2seq"
    adapter.build_seq2seq_profile(mapping, artifact, "test")
    compact = MT5ForConditionalGeneration.from_pretrained(artifact / "model")
    result = compare_teacher_forcing(original, compact, [3, 8, 1], [4, 1], mapping)
    assert result["retained_raw_logits"]["maximum_absolute_difference"] == 0.0
    assert result["source_covered"] and result["target_covered"]


def test_paired_bootstrap_is_deterministic() -> None:
    arguments = {
        "maximum_allowed_drop": 0.05,
        "iterations": 500,
        "seed": 7,
    }
    first = paired_bootstrap_non_inferiority([0.8, 0.9, 0.7], [0.8, 0.89, 0.72], **arguments)
    second = paired_bootstrap_non_inferiority([0.8, 0.9, 0.7], [0.8, 0.89, 0.72], **arguments)
    assert first == second
    assert first["non_inferiority_passed"]


def test_external_hook_secret_redaction() -> None:
    assert redact_secrets("api_key=abc123 token: xyz") == (
        "api_key=[REDACTED_SECRET] token: [REDACTED_SECRET]"
    )
