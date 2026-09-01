from pathlib import Path

import pytest
import torch

from vocabcraft.exceptions import ArtifactError, PackConflictError
from vocabcraft.mappings import IdMapping
from vocabcraft.packs import (
    build_cold_pack,
    load_pack,
    merge_packs,
    reconstruct_vocabulary_tensors,
)


def _state(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "shared.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset,
        "lm_head.weight": torch.arange(12, 24, dtype=torch.float32).reshape(4, 3) + offset,
        "block.weight": torch.eye(3),
    }


def _build(
    tmp_path: Path,
    name: str,
    ids: list[int],
    *,
    offset: float = 0.0,
) -> Path:
    destination = tmp_path / name
    build_cold_pack(
        _state(offset),
        ids,
        4,
        destination,
        source_model_hash="model-hash",
        tokenizer_hash="tokenizer-hash",
        source_revision="fixture",
        profile_id="test",
    )
    return destination


def test_cold_pack_reconstruction_restores_exact_rows(tmp_path: Path) -> None:
    original = _state()
    mapping = IdMapping.from_retained([0, 2], 4)
    pack = load_pack(_build(tmp_path, "cold", [1, 3]))
    compact = {
        "shared.weight": original["shared.weight"][[0, 2]],
        "lm_head.weight": original["lm_head.weight"][[0, 2]],
        "block.weight": original["block.weight"],
    }
    reconstructed = reconstruct_vocabulary_tensors(compact, mapping, pack)
    assert set(reconstructed) == set(original)
    assert all(torch.equal(reconstructed[name], tensor) for name, tensor in original.items())


def test_pack_merge_is_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path, "first", [3])
    second = _build(tmp_path, "second", [1])
    left = tmp_path / "left"
    right = tmp_path / "right"
    merge_packs([first, second], left)
    merge_packs([second, first], right)
    left_pack = load_pack(left)
    right_pack = load_pack(right)
    assert left_pack.original_ids == right_pack.original_ids == (1, 3)
    assert all(
        torch.equal(left_pack.tensors[name], right_pack.tensors[name]) for name in left_pack.tensors
    )


def test_conflicting_duplicate_pack_rows_fail(tmp_path: Path) -> None:
    first = _build(tmp_path, "first", [1])
    conflict = _build(tmp_path, "conflict", [1], offset=1.0)
    with pytest.raises(PackConflictError, match="conflicting row"):
        merge_packs([first, conflict], tmp_path / "merged")


def test_pack_hash_mismatch_fails(tmp_path: Path) -> None:
    pack = _build(tmp_path, "pack", [1])
    rows = pack / "rows.safetensors"
    rows.write_bytes(rows.read_bytes() + b"corruption")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_pack(pack)


def test_incomplete_reconstruction_fails(tmp_path: Path) -> None:
    mapping = IdMapping.from_retained([0], 4)
    pack = load_pack(_build(tmp_path, "cold", [1, 2]))
    compact = {
        "shared.weight": _state()["shared.weight"][[0]],
        "lm_head.weight": _state()["lm_head.weight"][[0]],
        "block.weight": _state()["block.weight"],
    }
    with pytest.raises(PackConflictError, match="missing original IDs"):
        reconstruct_vocabulary_tensors(compact, mapping, pack)
