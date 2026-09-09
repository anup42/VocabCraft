import json
import os
import sys
from pathlib import Path

import pytest

from vocabcraft.evaluation.external_hook import run_external_evaluation
from vocabcraft.exceptions import ValidationFailure


def test_preexisting_unchanged_metrics_are_rejected_and_preserved(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    original_bytes = b'{"task_score": 0.99}'
    metrics.write_bytes(original_bytes)
    with pytest.raises(ValidationFailure, match="stale metrics"):
        run_external_evaluation(
            [sys.executable, "-c", "pass"],
            model_path=tmp_path,
            metrics_file=metrics,
            timeout_seconds=10,
        )
    assert metrics.read_bytes() == original_bytes


def test_missing_metrics_do_not_pass_a_successful_command(tmp_path: Path) -> None:
    with pytest.raises(ValidationFailure, match="did not produce"):
        run_external_evaluation(
            [sys.executable, "-c", "pass"],
            model_path=tmp_path,
            metrics_file=tmp_path / "metrics.json",
            timeout_seconds=10,
        )


def test_placeholder_isolates_repeated_evaluations_and_preserves_declared_file(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.json"
    original_bytes = b'{"task_score": 0.25}'
    metrics.write_bytes(original_bytes)
    command = [
        sys.executable,
        "-c",
        "import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text("
        "json.dumps({'task_score': 0.8, 'model': sys.argv[2]}), encoding='utf-8')",
        "{metrics_file}",
        "{model_path}",
    ]
    runs = [
        run_external_evaluation(
            command,
            model_path=model,
            metrics_file=metrics,
            timeout_seconds=10,
        )
        for model in ["original-checkpoint", "compact-checkpoint"]
    ]
    assert runs[0]["metrics_file"] != runs[1]["metrics_file"]
    assert runs[0]["metrics"]["model"] == "original-checkpoint"
    assert runs[1]["metrics"]["model"] == "compact-checkpoint"
    assert all(run["metrics_freshness_verified"] for run in runs)
    assert all(run["temporary_metrics_file"] for run in runs)
    assert metrics.read_bytes() == original_bytes


def test_placeholder_failure_does_not_replace_declared_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"task_score": 0.9}', encoding="utf-8")
    original_bytes = metrics.read_bytes()
    with pytest.raises(ValidationFailure, match="external evaluation failed"):
        run_external_evaluation(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}'); sys.exit(1)",
                "{metrics_file}",
            ],
            model_path=tmp_path,
            metrics_file=metrics,
            timeout_seconds=10,
        )
    assert metrics.read_bytes() == original_bytes


def test_legacy_changed_metrics_are_accepted(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"task_score": 0.1}', encoding="utf-8")
    result = run_external_evaluation(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{\"task_score\":0.8}')",
            str(metrics),
        ],
        model_path=tmp_path,
        metrics_file=metrics,
        timeout_seconds=10,
    )
    assert result["metrics"] == {"task_score": 0.8}
    assert result["temporary_metrics_file"] is False


def test_legacy_same_metrics_rewritten_are_fresh(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"task_score": 0.8}', encoding="utf-8")
    os.utime(metrics, (1_000_000, 1_000_000))
    result = run_external_evaluation(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; path=pathlib.Path(sys.argv[1]); "
            "path.write_bytes(path.read_bytes())",
            str(metrics),
        ],
        model_path=tmp_path,
        metrics_file=metrics,
        timeout_seconds=10,
    )
    assert result["metrics"] == {"task_score": 0.8}


@pytest.mark.parametrize("payload", ["null", "[]", "invalid-json"])
def test_metrics_must_be_valid_json_object(tmp_path: Path, payload: str) -> None:
    with pytest.raises(ValidationFailure, match=r"metrics.*JSON"):
        run_external_evaluation(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                "{metrics_file}",
                payload,
            ],
            model_path=tmp_path,
            metrics_file=tmp_path / "metrics.json",
            timeout_seconds=10,
        )


def test_failed_command_redacts_secrets_even_when_metrics_exist(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"task_score": 1}), encoding="utf-8")
    with pytest.raises(ValidationFailure) as error:
        run_external_evaluation(
            [sys.executable, "-c", "import sys; print('token=fixture-secret'); sys.exit(1)"],
            model_path=tmp_path,
            metrics_file=metrics,
            timeout_seconds=10,
        )
    assert "fixture-secret" not in str(error.value)
    assert "REDACTED_SECRET" in str(error.value)
