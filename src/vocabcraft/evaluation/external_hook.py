"""Local-only external evaluation hook with secret-redacted reporting."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from vocabcraft.exceptions import ValidationFailure

_SECRET = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)([^\s,;]+)")


def redact_secrets(text: str) -> str:
    """Redact common credential assignments from captured process output."""

    return _SECRET.sub(r"\1\2[REDACTED_SECRET]", text)


def run_external_evaluation(
    command: list[str],
    *,
    model_path: str | Path,
    metrics_file: str | Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run a local command and load only its declared metrics JSON output."""

    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    candidate_model = Path(model_path)
    resolved_model = str(candidate_model.resolve()) if candidate_model.exists() else str(model_path)
    expanded = [argument.replace("{model_path}", resolved_model) for argument in command]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(term in key.upper() for term in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
    }
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
        raise ValidationFailure(f"external evaluation timed out after {timeout_seconds}s") from exc
    metrics_path = Path(metrics_file)
    metrics: Any = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationFailure(f"cannot read external metrics JSON: {exc}") from exc
    result = {
        "command": [redact_secrets(value) for value in expanded],
        "exit_status": completed.returncode,
        "stdout": redact_secrets(completed.stdout),
        "stderr": redact_secrets(completed.stderr),
        "metrics": metrics,
        "metrics_file": str(metrics_path.resolve()),
        "local_only": True,
    }
    if completed.returncode != 0:
        raise ValidationFailure(
            "external evaluation failed: " + json.dumps(result, ensure_ascii=False)
        )
    if metrics is None:
        raise ValidationFailure("external evaluation did not produce its declared metrics file")
    return result
