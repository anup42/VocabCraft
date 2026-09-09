import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from vocabcraft import cli, workflows
from vocabcraft.config import VocabCraftConfig, load_config
from vocabcraft.mappings import IdMapping


def _prepare_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    *,
    mode: str,
    records: list[dict[str, Any]],
    teacher_pass: bool = True,
    exact_match: bool = True,
    require_exact: bool = False,
) -> tuple[VocabCraftConfig, Path, Path]:
    """Mock model execution while retaining real coverage, aggregation, reports and CLI."""

    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "project: {name: VocabCraft}\nmodel: {id: fixture}\n"
        "profile: {id: test, description: test, languages: [en, hi], unicode_scripts: [Latin]}\n"
        f"validation: {{generation_exact_match_required: {str(require_exact).lower()}}}\n",
        encoding="utf-8",
    )
    config = load_config(profile)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 8], 11)
    metadata = {
        "model_family": "mt5",
        "vocabcraft_mode": mode,
        "compact_vocabulary_size": len(mapping.new_to_old),
        "original_model_vocabulary_size": 11,
        "tokenizer_sha256": "fixture-tokenizer",
        "model_state_sha256": "fixture-model",
        "original_decoder_start_token_id": 0,
        "original_eos_token_id": 1,
    }
    (artifact / "vocabcraft-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact / "mapping-report.json").write_text(json.dumps(mapping.to_dict()), encoding="utf-8")
    adapter = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="mt5")),
        tokenizer=tokenizer,
        tokenizer_hash=lambda: "fixture-tokenizer",
        model_state_hash=lambda: "fixture-model",
    )
    monkeypatch.setattr(workflows, "load_adapter", lambda *args, **kwargs: adapter)
    monkeypatch.setattr(workflows, "load_encoder_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        workflows, "load_compact_mt5_seq2seq_model", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(workflows, "load_guarded_mt5_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        workflows,
        "compare_encoder_models",
        lambda *args, **kwargs: {
            "per_example": [{"passed": True} for _ in range(args[2].shape[0])]
        },
    )
    monkeypatch.setattr(
        workflows, "compare_teacher_forcing", lambda *args, **kwargs: {"passed": teacher_pass}
    )
    monkeypatch.setattr(
        workflows,
        "compare_greedy_generation",
        lambda *args, **kwargs: {
            "exact_original_token_id_match": exact_match,
            "exact_decoded_text_match": exact_match,
        },
    )
    monkeypatch.setattr(
        workflows,
        "GuardedMT5Generator",
        lambda *args, **kwargs: SimpleNamespace(
            generate_greedy=lambda *args: {
                "generated": False,
                "fallback_occurred": True,
                "unsupported_profile": True,
            }
        ),
    )
    return config, artifact, data


@pytest.mark.parametrize(
    "mode,records,teacher_pass,exact_match,require_exact,failure_key",
    [
        (
            "seq2seq_compact", [{"id": "teacher", "source": "hello", "target": "फोन"}],
            False, True, False, "teacher_forcing_failed",
        ),
        (
            "seq2seq_compact", [{"id": "generation", "text": "hello"}],
            True, False, True, "generation_exact_match_failed",
        ),
        ("encoder_exact", [], True, True, False, "empty_validation_data"),
        (
            "encoder_exact", [{"id": "fallback", "text": "東京"}],
            True, True, False, "no_mode_comparisons",
        ),
        (
            "seq2seq_guarded", [{"id": "unsupported", "text": "東京"}],
            True, True, False, "guarded_generation_unsupported",
        ),
    ],
)
def test_validation_workflow_and_cli_reject_failed_mode_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_tokenizer: Any,
    mode: str,
    records: list[dict[str, Any]],
    teacher_pass: bool,
    exact_match: bool,
    require_exact: bool,
    failure_key: str,
) -> None:
    config, artifact, data = _prepare_workflow(
        tmp_path,
        monkeypatch,
        fake_tokenizer,
        mode=mode,
        records=records,
        teacher_pass=teacher_pass,
        exact_match=exact_match,
        require_exact=require_exact,
    )
    report = workflows.validate_to_directory("fixture", artifact, config, data, tmp_path / "report")
    assert report["passed"] is False
    assert report["strict_failures"][failure_key] is True
    saved = json.loads((tmp_path / "report" / "validation.json").read_text(encoding="utf-8"))
    assert saved["passed"] is False
    assert saved["strict_failures"][failure_key] is True

    result = CliRunner().invoke(
        cli.app,
        [
            "validate", "--original", "fixture", "--compact", str(artifact),
            "--profile", str(config.source_path), "--data", str(data),
            "--output", str(tmp_path / "cli-report"),
        ],
    )
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["result"]["strict_failures"][failure_key] is True


def test_covered_encoder_validation_can_pass_with_fallback_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_tokenizer: Any
) -> None:
    config, artifact, data = _prepare_workflow(
        tmp_path,
        monkeypatch,
        fake_tokenizer,
        mode="encoder_exact",
        records=[
            {"id": "covered", "text": "hello"},
            {"id": "fallback", "text": "東京"},
        ],
    )
    result = workflows.validate_to_directory("fixture", artifact, config, data, tmp_path / "report")
    assert result["passed"] is True
    assert result["coverage"]["fallback_examples"] == 1
    assert result["comparison_counts"]["encoder"] == 1
