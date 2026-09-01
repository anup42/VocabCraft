"""Atomic artifact-directory and serialization utilities."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vocabcraft.exceptions import ArtifactError


@contextmanager
def atomic_output_directory(destination: str | Path) -> Iterator[Path]:
    """Stage output beside its destination and publish it only after success."""

    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ArtifactError(f"refusing to overwrite existing artifact directory: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield staging
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_json(path: str | Path, value: Any) -> None:
    """Write readable, deterministic UTF-8 JSON."""

    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_json(path: str | Path) -> Any:
    """Read JSON while preserving structured parse failures."""

    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {source}: {exc}") from exc
