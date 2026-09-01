"""Streaming calibration dataset support."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from vocabcraft.exceptions import ConfigurationError


@dataclass(frozen=True)
class CalibrationRecord:
    """A normalized encoder or sequence-to-sequence calibration record."""

    id: str
    source: str
    target: str | None
    source_language: str | None
    target_language: str | None
    domain: str | None
    critical: bool
    metadata: dict[str, Any]


def _normalize_record(raw: dict[str, Any], path: Path, line_number: int) -> CalibrationRecord:
    record_id = raw.get("id")
    source = raw.get("source", raw.get("text"))
    target = raw.get("target")
    if not isinstance(record_id, str) or not record_id:
        raise ConfigurationError(f"{path}:{line_number}: record id must be a non-empty string")
    if not isinstance(source, str) or not source:
        raise ConfigurationError(f"{path}:{line_number}: text or source must be non-empty")
    if target is not None and not isinstance(target, str):
        raise ConfigurationError(f"{path}:{line_number}: target must be a string when present")
    language = raw.get("source_language", raw.get("language"))
    target_language = raw.get("target_language")
    domain = raw.get("domain")
    string_fields = (
        (language, "language"),
        (target_language, "target_language"),
        (domain, "domain"),
    )
    for value, name in string_fields:
        if value is not None and not isinstance(value, str):
            raise ConfigurationError(f"{path}:{line_number}: {name} must be a string")
    return CalibrationRecord(
        id=record_id,
        source=source,
        target=target,
        source_language=cast(str | None, language),
        target_language=cast(str | None, target_language),
        domain=cast(str | None, domain),
        critical=bool(raw.get("critical", False)),
        metadata=dict(raw),
    )


def stream_jsonl(path: str | Path) -> Iterator[CalibrationRecord]:
    """Yield normalized calibration records one line at a time."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw_object = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(
                        f"{source}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(raw_object, dict):
                    raise ConfigurationError(f"{source}:{line_number}: record must be an object")
                yield _normalize_record(cast(dict[str, Any], raw_object), source, line_number)
    except OSError as exc:
        raise ConfigurationError(f"cannot read calibration file {source}: {exc}") from exc
