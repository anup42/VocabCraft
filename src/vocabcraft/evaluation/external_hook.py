"""Local-only external evaluation hook with secret-redacted reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from vocabcraft.exceptions import ValidationFailure

_SECRET = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)([^\s,;]+)")


def redact_secrets(text: str) -> str:
    """Redact common credential assignments from captured process output."""

    return _SECRET.sub(r"\1\2[REDACTED_SECRET]", text)


def _metrics_fingerprint(path: Path) -> tuple[int, int, int, int, int, str] | None:
    """Observe a legacy output without removing or modifying a preexisting file."""

    if not path.exists():
        return None
    if not path.is_file():
        raise ValidationFailure("external metrics path must be a regular file")
    try:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValidationFailure(f"cannot inspect external metrics file: {exc}") from exc
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, digest


def run_external_evaluation(
    command: list[str],
    *,
    model_path: str | Path,
    metrics_file: str | Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run a local command and accept only freshly produced metrics.

    Commands can use ``{model_path}`` and ``{metrics_file}`` placeholders. The
    latter receives a unique temporary output path for each invocation, keeping
    the configured metrics file untouched even if the evaluator fails. Legacy
    commands writing a fixed configured path must change that file's metadata
    or content during this invocation; an unchanged previous result is rejected.
    """

    if not command:
        raise ValueError("external evaluation command cannot be empty")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    candidate_model = Path(model_path)
    resolved_model = str(candidate_model.resolve()) if candidate_model.exists() else str(model_path)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(term in key.upper() for term in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
    }
    declared_metrics = Path(metrics_file).resolve()
    isolated_output = any("{metrics_file}" in argument for argument in command)
    context = (
        TemporaryDirectory(prefix="vocabcraft-evaluation-") if isolated_output else nullcontext()
    )
    with context as temporary_root:
        metrics_path = (
            Path(temporary_root) / "metrics.json"
            if temporary_root is not None
            else declared_metrics
        )
        before = _metrics_fingerprint(metrics_path)
        expanded = [
            argument.replace("{model_path}", resolved_model).replace(
                "{metrics_file}", str(metrics_path)
            )
            for argument in command
        ]
        try:
            completed = subprocess.run(
                expanded,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValidationFailure(
                f"external evaluation timed out after {timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise ValidationFailure(f"cannot start external evaluation: {exc}") from exc
        result: dict[str, Any] = {
            "command": [redact_secrets(value) for value in expanded],
            "exit_status": completed.returncode,
            "stdout": redact_secrets(completed.stdout),
            "stderr": redact_secrets(completed.stderr),
            "metrics_file": str(metrics_path),
            "declared_metrics_file": str(declared_metrics),
            "temporary_metrics_file": isolated_output,
            "local_only": True,
        }
        if completed.returncode != 0:
            raise ValidationFailure(
                "external evaluation failed: " + json.dumps(result, ensure_ascii=False)
            )
        after = _metrics_fingerprint(metrics_path)
        if after is None:
            raise ValidationFailure("external evaluation did not produce its declared metrics file")
        if before is not None and before == after:
            raise ValidationFailure(
                "external evaluation left its metrics file unchanged; stale metrics are rejected. "
                "Use {metrics_file} in the command for isolated per-run output."
            )
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValidationFailure(f"cannot read external metrics JSON: {exc}") from exc
        if not isinstance(metrics, dict):
            raise ValidationFailure("external evaluation metrics must be a JSON object")
        result["metrics"] = metrics
        result["metrics_freshness_verified"] = True
        return result
