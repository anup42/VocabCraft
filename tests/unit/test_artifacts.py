from pathlib import Path

import pytest

from vocabcraft.artifacts import atomic_output_directory, read_json, write_json
from vocabcraft.exceptions import ArtifactError


def test_atomic_artifact_creation(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    with atomic_output_directory(destination) as staging:
        write_json(staging / "metadata.json", {"complete": True})
        assert not destination.exists()
    assert read_json(destination / "metadata.json") == {"complete": True}


def test_atomic_artifact_failure_cleans_staging(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    with pytest.raises(RuntimeError), atomic_output_directory(destination):
        raise RuntimeError("simulated failure")
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_artifact_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    destination.mkdir()
    with (
        pytest.raises(ArtifactError, match="refusing to overwrite"),
        atomic_output_directory(destination),
    ):
        pass
