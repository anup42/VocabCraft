"""Cold/language pack export, deterministic merging, and exact reconstruction."""

from __future__ import annotations

import copy
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    MT5Config,
    MT5EncoderModel,
    MT5ForConditionalGeneration,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from vocabcraft import __version__
from vocabcraft.artifacts import atomic_output_directory, read_json, write_json
from vocabcraft.exceptions import ArtifactError, PackConflictError, ValidationFailure
from vocabcraft.hashing import sha256_bytes, sha256_file
from vocabcraft.mappings import IdMapping


@dataclass(frozen=True)
class LoadedPack:
    """Verified pack contents."""

    original_ids: tuple[int, ...]
    tensors: dict[str, torch.Tensor]
    manifest: dict[str, Any]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return sha256_bytes(value.numpy().tobytes())


def _vocabulary_rows(
    state_dict: dict[str, torch.Tensor],
    original_ids: tuple[int, ...],
    vocabulary_size: int,
    vocabulary_tensor_names: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index = torch.tensor(original_ids, dtype=torch.long)
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    if vocabulary_tensor_names is not None and set(vocabulary_tensor_names) - set(state_dict):
        raise ArtifactError("declared vocabulary tensor is missing from source state")
    for name, tensor in sorted(state_dict.items()):
        if vocabulary_tensor_names is not None:
            if name not in vocabulary_tensor_names:
                continue
            if tensor.ndim == 0 or tensor.shape[0] != vocabulary_size:
                raise ArtifactError(f"declared vocabulary tensor {name} has no leading vocab axis")
            axes = [0]
        else:
            axes = [axis for axis, size in enumerate(tensor.shape) if size == vocabulary_size]
        if not axes:
            continue
        if axes != [0]:
            raise ArtifactError(
                f"cannot pack vocabulary tensor {name} with non-leading axes {axes}"
            )
        rows = tensor.detach().cpu().index_select(0, index).contiguous()
        tensors[name] = rows
        metadata[name] = {
            "source_shape": list(tensor.shape),
            "pack_shape": list(rows.shape),
            "dtype": str(tensor.dtype),
        }
    if not tensors:
        raise ArtifactError("state dictionary contains no vocabulary-sized tensors")
    return tensors, metadata


def _write_pack(
    staging: Path,
    original_ids: tuple[int, ...],
    tensors: dict[str, torch.Tensor],
    manifest: dict[str, Any],
) -> None:
    save_file(tensors, staging / "rows.safetensors")
    write_json(staging / "original-ids.json", list(original_ids))
    checksums = {
        "rows.safetensors": sha256_file(staging / "rows.safetensors"),
        "tensors": {name: _tensor_sha256(tensor) for name, tensor in sorted(tensors.items())},
    }
    write_json(staging / "manifest.json", manifest)
    write_json(staging / "checksums.json", checksums)


def build_cold_pack(
    state_dict: dict[str, torch.Tensor],
    original_ids: list[int] | tuple[int, ...],
    original_vocab_size: int,
    destination: str | Path,
    *,
    source_model_hash: str,
    tokenizer_hash: str,
    source_revision: str | None,
    profile_id: str,
    vocabulary_tensor_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export exact excluded rows and compatibility metadata atomically."""

    ordered = tuple(sorted(original_ids))
    if len(set(ordered)) != len(ordered):
        raise ArtifactError("pack original IDs must be unique")
    if any(token_id < 0 or token_id >= original_vocab_size for token_id in ordered):
        raise ArtifactError("pack original ID is outside the source vocabulary")
    tensors, tensor_metadata = _vocabulary_rows(
        state_dict, ordered, original_vocab_size, vocabulary_tensor_names
    )
    manifest: dict[str, Any] = {
        "format_version": 1,
        "pack_type": "cold_fallback",
        "source_model_sha256": source_model_hash,
        "tokenizer_sha256": tokenizer_hash,
        "source_revision": source_revision,
        "original_vocab_size": original_vocab_size,
        "original_ids": list(ordered),
        "tensor_names": sorted(tensors),
        "tensors": tensor_metadata,
        "profile_id": profile_id,
        "vocabcraft_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with atomic_output_directory(destination) as staging:
        _write_pack(staging, ordered, tensors, manifest)
    return manifest


def load_pack(path: str | Path) -> LoadedPack:
    """Load a pack and verify file/tensor checksums before returning rows."""

    root = Path(path)
    manifest_object = read_json(root / "manifest.json")
    ids_object = read_json(root / "original-ids.json")
    checksums_object = read_json(root / "checksums.json")
    if not isinstance(manifest_object, dict) or not isinstance(checksums_object, dict):
        raise ArtifactError("pack manifest/checksums must be JSON objects")
    if not isinstance(ids_object, list) or not all(isinstance(value, int) for value in ids_object):
        raise ArtifactError("pack original-ids.json must be an integer list")
    expected_file_hash = checksums_object.get("rows.safetensors")
    actual_file_hash = sha256_file(root / "rows.safetensors")
    if expected_file_hash != actual_file_hash:
        raise ArtifactError("pack rows.safetensors checksum mismatch")
    tensors = load_file(root / "rows.safetensors", device="cpu")
    expected_tensors = checksums_object.get("tensors")
    if not isinstance(expected_tensors, dict):
        raise ArtifactError("pack tensor checksums are missing")
    for name, tensor in tensors.items():
        if expected_tensors.get(name) != _tensor_sha256(tensor):
            raise ArtifactError(f"pack tensor checksum mismatch: {name}")
    original_ids = tuple(ids_object)
    vocabulary_size = manifest_object.get("original_vocab_size")
    if (
        type(vocabulary_size) is not int
        or vocabulary_size < 1
        or tuple(sorted(set(original_ids))) != original_ids
        or any(type(value) is not int or not 0 <= value < vocabulary_size for value in original_ids)
    ):
        raise ArtifactError("pack IDs must be sorted, unique, and within the original vocabulary")
    if set(tensors) != set(expected_tensors) or sorted(tensors) != manifest_object.get(
        "tensor_names"
    ):
        raise ArtifactError("pack tensor names disagree with manifest/checksums")
    if manifest_object.get("original_ids") != list(original_ids):
        raise ArtifactError("pack manifest and original-ids.json disagree")
    for name, tensor in tensors.items():
        if tensor.ndim == 0 or tensor.shape[0] != len(original_ids):
            raise ArtifactError(f"pack tensor {name} row count does not match ID count")
    return LoadedPack(original_ids, tensors, manifest_object)


def _assert_pack_compatible(reference: LoadedPack, candidate: LoadedPack) -> None:
    for field in ("source_model_sha256", "tokenizer_sha256", "original_vocab_size"):
        if reference.manifest.get(field) != candidate.manifest.get(field):
            raise PackConflictError(f"incompatible pack {field}")
    if set(reference.tensors) != set(candidate.tensors):
        raise PackConflictError("packs contain different vocabulary tensor sets")
    for name in reference.tensors:
        if reference.tensors[name].shape[1:] != candidate.tensors[name].shape[1:]:
            raise PackConflictError(f"incompatible row shape for tensor {name}")
        if reference.tensors[name].dtype != candidate.tensors[name].dtype:
            raise PackConflictError(f"incompatible dtype for tensor {name}")


def merge_packs(pack_paths: Sequence[str | Path], destination: str | Path) -> dict[str, Any]:
    """Merge packs by original ID in deterministic order; conflicting rows fail."""

    if not pack_paths:
        raise ArtifactError("at least one pack is required")
    loaded = [load_pack(path) for path in sorted(pack_paths, key=lambda item: str(item))]
    reference = loaded[0]
    rows: dict[str, dict[int, torch.Tensor]] = {name: {} for name in reference.tensors}
    for pack in loaded:
        _assert_pack_compatible(reference, pack)
        for row_index, original_id in enumerate(pack.original_ids):
            for name, tensor in pack.tensors.items():
                candidate = tensor[row_index]
                existing = rows[name].get(original_id)
                if existing is not None and not torch.equal(existing, candidate):
                    raise PackConflictError(
                        f"conflicting row data for tensor {name}, original ID {original_id}"
                    )
                rows[name][original_id] = candidate.clone()
    merged_ids = tuple(sorted({token_id for values in rows.values() for token_id in values}))
    for name, values in rows.items():
        if set(values) != set(merged_ids):
            raise PackConflictError(f"tensor {name} does not cover every merged ID")
    tensors = {
        name: (
            torch.stack([values[token_id] for token_id in merged_ids])
            if merged_ids
            else reference.tensors[name][:0].clone()
        )
        for name, values in sorted(rows.items())
    }
    manifest = dict(reference.manifest)
    manifest.update(
        {
            "pack_type": "merged",
            "original_ids": list(merged_ids),
            "source_pack_count": len(loaded),
            "profile_id": "merged",
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    for name, tensor in tensors.items():
        source_shape = list(manifest["tensors"][name]["source_shape"])
        manifest["tensors"][name] = {
            "source_shape": source_shape,
            "pack_shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    with atomic_output_directory(destination) as staging:
        _write_pack(staging, merged_ids, tensors, manifest)
    return manifest


def reconstruct_vocabulary_tensors(
    compact_state: dict[str, torch.Tensor], mapping: IdMapping, pack: LoadedPack
) -> dict[str, torch.Tensor]:
    """Restore every vocabulary-sized tensor, requiring complete exact row coverage."""

    if pack.manifest.get("original_vocab_size") != mapping.original_vocab_size:
        raise PackConflictError("pack vocabulary size does not match mapping")
    retained_ids = set(mapping.new_to_old)
    cold_ids = set(pack.original_ids)
    expected_ids = set(range(mapping.original_vocab_size))
    if retained_ids & cold_ids:
        raise PackConflictError("retained and pack IDs overlap")
    if retained_ids | cold_ids != expected_ids:
        missing = sorted(expected_ids - retained_ids - cold_ids)
        raise PackConflictError(f"reconstruction is missing original IDs: {missing[:20]}")
    result: dict[str, torch.Tensor] = {}
    retained_index = torch.tensor(mapping.new_to_old, dtype=torch.long)
    cold_index = torch.tensor(pack.original_ids, dtype=torch.long)
    for name, compact_tensor in compact_state.items():
        packed_tensor = pack.tensors.get(name)
        if packed_tensor is None:
            result[name] = compact_tensor.detach().clone()
            continue
        if compact_tensor.shape[0] == mapping.original_vocab_size:
            if not torch.equal(
                compact_tensor.detach().cpu().index_select(0, cold_index),
                packed_tensor.to(dtype=compact_tensor.dtype),
            ):
                raise PackConflictError(
                    f"already-full compact tensor {name} conflicts with packed cold rows"
                )
            result[name] = compact_tensor.detach().clone()
            continue
        if compact_tensor.shape[0] != len(mapping.new_to_old):
            raise PackConflictError(f"compact tensor {name} row count does not match mapping")
        if compact_tensor.shape[1:] != packed_tensor.shape[1:]:
            raise PackConflictError(f"row shape mismatch for reconstructed tensor {name}")
        restored = torch.empty(
            (mapping.original_vocab_size, *compact_tensor.shape[1:]),
            dtype=compact_tensor.dtype,
        )
        restored.index_copy_(0, retained_index, compact_tensor.detach().cpu())
        restored.index_copy_(0, cold_index, packed_tensor.to(dtype=compact_tensor.dtype))
        result[name] = restored
    missing_tensor_names = set(pack.tensors) - set(compact_state)
    if missing_tensor_names:
        raise PackConflictError(
            f"compact checkpoint is missing packed tensors: {sorted(missing_tensor_names)}"
        )
    return result


def reconstruct_full_artifact(
    compact_artifact: str | Path,
    pack_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Rebuild a reloadable original-vocabulary model artifact offline."""

    root = Path(compact_artifact)
    metadata = read_json(root / "vocabcraft-metadata.json")
    mapping_payload = read_json(root / "mapping-report.json")
    source_config_payload = read_json(root / "source-config.json")
    if not isinstance(metadata, dict) or not isinstance(mapping_payload, dict):
        raise ArtifactError("compact metadata and mapping must be JSON objects")
    if not isinstance(source_config_payload, dict):
        raise ArtifactError("source-config.json must be a JSON object")
    mapping = IdMapping.from_dict(mapping_payload)
    pack = load_pack(pack_path)
    if pack.manifest.get("source_model_sha256") != metadata.get("model_state_sha256"):
        raise PackConflictError("pack source-model hash does not match compact artifact")
    if pack.manifest.get("tokenizer_sha256") != metadata.get("tokenizer_sha256"):
        raise PackConflictError("pack tokenizer hash does not match compact artifact")
    mode = metadata.get("vocabcraft_mode")
    from vocabcraft.models.registry import artifact_family, load_encoder_model

    family = artifact_family(metadata)
    compact_model: Any
    original_model: Any
    source_config: Any
    if family == "xlm-roberta":
        if mode != "encoder_exact":
            raise ArtifactError("XLM-R reconstruction supports only encoder_exact")
        source_config = XLMRobertaConfig.from_dict(source_config_payload)
        compact_model = load_encoder_model(root, metadata)
        original_model = XLMRobertaModel(
            source_config, add_pooling_layer=bool(metadata.get("add_pooling_layer", False))
        )
    elif mode == "encoder_exact":
        source_config = MT5Config.from_dict(source_config_payload)
        source_config.tie_word_embeddings = bool(
            source_config_payload.get("tie_word_embeddings", True)
        )
        compact_model = load_encoder_model(root, metadata)
        original_model = MT5EncoderModel(source_config)
    elif mode == "seq2seq_compact":
        from vocabcraft.models.mt5_seq2seq import load_compact_mt5_seq2seq_model

        source_config = MT5Config.from_dict(source_config_payload)
        source_config.tie_word_embeddings = bool(
            source_config_payload.get("tie_word_embeddings", True)
        )
        compact_model = load_compact_mt5_seq2seq_model(root / "model")
        from vocabcraft.models.mt5 import MT5Adapter

        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", use_fast=False)
        adapter = MT5Adapter(compact_model, tokenizer, "reconstruction")
        original_model = adapter._new_compact_seq2seq_model(copy.deepcopy(source_config))
    elif mode == "seq2seq_guarded":
        from vocabcraft.models.mt5_guarded_generation import load_guarded_mt5_model

        source_config = MT5Config.from_dict(source_config_payload)
        source_config.tie_word_embeddings = False
        compact_model = load_guarded_mt5_model(root / "model", mapping.original_vocab_size)
        original_model = MT5ForConditionalGeneration(source_config)
    else:
        raise ArtifactError(f"unsupported compact artifact mode for reconstruction: {mode!r}")
    restored_state = reconstruct_vocabulary_tensors(compact_model.state_dict(), mapping, pack)
    original_model.to(dtype=next(compact_model.parameters()).dtype)
    original_model.load_state_dict(restored_state, strict=True)
    generation_path = root / "source-generation-config.json"
    if mode != "encoder_exact" and generation_path.is_file():
        generation_payload = read_json(generation_path)
        if not isinstance(generation_payload, dict):
            raise ArtifactError("source generation configuration must be a JSON object")
        original_model.generation_config = GenerationConfig.from_dict(generation_payload)
    del compact_model
    gc.collect()
    with atomic_output_directory(destination) as staging:
        model_dir = staging / "model"
        original_model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer", use_fast=family == "xlm-roberta"
        )
        tokenizer.save_pretrained(staging / "tokenizer")
        sentence_config = root / "sentence-embedding-config.json"
        if sentence_config.is_file():
            write_json(staging / sentence_config.name, read_json(sentence_config))
        result = {
            "format_version": 1,
            "source_compact_artifact": str(root.resolve()),
            "source_pack": str(Path(pack_path).resolve()),
            "mode": mode,
            "model_family": family,
            "restored_vocabulary_size": mapping.original_vocab_size,
            "reconstruction_complete": True,
        }
        write_json(staging / "reconstruction.json", result)
        del original_model
        gc.collect()
        if mode == "encoder_exact":
            if family == "xlm-roberta":
                from vocabcraft.models.xlm_roberta import load_xlm_roberta_model

                reloaded_state = load_xlm_roberta_model(
                    model_dir, add_pooling_layer=bool(metadata.get("add_pooling_layer", False))
                ).state_dict()
            else:
                reloaded_state = MT5EncoderModel.from_pretrained(model_dir).state_dict()
        else:
            from vocabcraft.models.mt5_seq2seq import load_compact_mt5_seq2seq_model

            reloaded_state = load_compact_mt5_seq2seq_model(model_dir).state_dict()
        if set(reloaded_state) != set(restored_state) or any(
            not torch.equal(reloaded_state[name], value) for name, value in restored_state.items()
        ):
            raise ValidationFailure("reloaded reconstructed model differs from restored state")
    return result
