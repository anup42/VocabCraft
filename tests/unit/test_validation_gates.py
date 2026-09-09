from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vocabcraft.config import ValidationConfig
from vocabcraft.evaluation.validation_gates import aggregate_validation_gates


def _row(comparison: dict[str, Any] | None) -> dict[str, Any]:
    return {"id": "sample", "comparison": comparison}


def test_encoder_gate_requires_data_and_actual_comparison() -> None:
    for total, rows in [(0, []), (1, []), (1, [_row(None)])]:
        result = aggregate_validation_gates(
            "encoder_exact",
            ValidationConfig(),
            total_examples=total,
            missing_critical_examples=[],
            new_unk_count=0,
            encoder_results=rows,
        )
        assert not result.passed
        assert result.strict_failures["no_mode_comparisons"]


def test_encoder_fallback_is_reported_without_hiding_real_comparison_failure() -> None:
    result = aggregate_validation_gates(
        "encoder_exact",
        ValidationConfig(),
        total_examples=2,
        missing_critical_examples=[],
        new_unk_count=0,
        encoder_results=[_row(None), _row({"passed": False})],
    )
    assert not result.passed
    assert result.strict_failures["encoder_equivalence_failed"]
    assert result.comparison_counts["encoder"] == 1


def test_teacher_forcing_failure_changes_aggregate_result() -> None:
    result = aggregate_validation_gates(
        "seq2seq_compact",
        ValidationConfig(),
        total_examples=1,
        missing_critical_examples=[],
        new_unk_count=0,
        teacher_results=[_row({"passed": False})],
        generation_results=[
            _row({"exact_original_token_id_match": True, "exact_decoded_text_match": True})
        ],
    )
    assert not result.passed
    assert result.strict_failures["teacher_forcing_failed"]


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("id_match,text_match", [(False, True), (True, False), (False, False)])
def test_compact_generation_exact_match_setting_is_enforced(
    required: bool, id_match: bool, text_match: bool
) -> None:
    result = aggregate_validation_gates(
        "seq2seq_compact",
        ValidationConfig(generation_exact_match_required=required),
        total_examples=1,
        missing_critical_examples=[],
        new_unk_count=0,
        generation_results=[
            _row(
                {"exact_original_token_id_match": id_match, "exact_decoded_text_match": text_match}
            )
        ],
    )
    assert result.passed is not required
    assert result.strict_failures["generation_exact_match_failed"] is required


def test_requested_generation_check_cannot_pass_using_teacher_forcing_alone() -> None:
    result = aggregate_validation_gates(
        "seq2seq_compact",
        ValidationConfig(generation_exact_match_required=True),
        total_examples=1,
        missing_critical_examples=[],
        new_unk_count=0,
        teacher_results=[_row({"passed": True})],
    )
    assert not result.passed
    assert result.strict_failures["generation_exact_match_failed"]


def test_guarded_unsupported_case_fails_even_with_successful_cases() -> None:
    result = aggregate_validation_gates(
        "seq2seq_guarded",
        ValidationConfig(),
        total_examples=2,
        missing_critical_examples=[],
        new_unk_count=0,
        generation_results=[
            _row({"guarded_result": {"generated": False, "unsupported_profile": True}}),
            _row(
                {
                    "guarded_result": {"generated": True},
                    "exact_original_token_id_match": True,
                }
            ),
        ],
    )
    assert not result.passed
    assert result.strict_failures["guarded_generation_unsupported"]
    assert result.comparison_counts["generation"] == 1


def test_guarded_divergence_fails_without_compact_generation_opt_in() -> None:
    result = aggregate_validation_gates(
        "seq2seq_guarded",
        ValidationConfig(generation_exact_match_required=False),
        total_examples=1,
        missing_critical_examples=[],
        new_unk_count=0,
        generation_results=[
            _row(
                {
                    "guarded_result": {"generated": True},
                    "exact_original_token_id_match": False,
                }
            )
        ],
    )
    assert not result.passed
    assert result.strict_failures["generation_exact_match_failed"]


def test_guarded_full_model_fallback_can_pass_when_verified() -> None:
    result = aggregate_validation_gates(
        "seq2seq_guarded",
        ValidationConfig(),
        total_examples=1,
        missing_critical_examples=[],
        new_unk_count=0,
        generation_results=[
            _row(
                {
                    "guarded_result": {"generated": True, "fallback_occurred": True},
                    "exact_original_token_id_match": True,
                }
            )
        ],
    )
    assert result.passed
    assert result.comparison_counts["guarded_fallback_generation"] == 1
    assert result.comparison_counts["guarded_compact_generation"] == 0


def test_coverage_thresholds_apply_alongside_actual_comparisons() -> None:
    result = aggregate_validation_gates(
        "encoder_exact",
        ValidationConfig(),
        total_examples=1,
        missing_critical_examples=["critical"],
        new_unk_count=1,
        encoder_results=[_row({"passed": True})],
    )
    assert not result.passed
    assert result.strict_failures["new_unk_threshold_exceeded"]
    assert result.strict_failures["missing_critical_threshold_exceeded"]


def test_validation_cli_uses_failed_aggregate_for_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vocabcraft import cli

    aggregate = aggregate_validation_gates(
        "seq2seq_compact",
        ValidationConfig(),
        total_examples=1,
        missing_critical_examples=[],
        new_unk_count=0,
        teacher_results=[_row({"passed": False})],
    )
    profile = tmp_path / "profile.yaml"
    data = tmp_path / "data.jsonl"
    artifact = tmp_path / "compact"
    profile.write_text("profile", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")
    artifact.mkdir()
    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(
        cli,
        "validate_to_directory",
        lambda *args: {"passed": aggregate.passed, "strict_failures": aggregate.strict_failures},
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "validate", "--original", "fixture", "--compact", str(artifact),
            "--profile", str(profile), "--data", str(data), "--output", str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 3
    assert '"teacher_forcing_failed": true' in result.stdout
