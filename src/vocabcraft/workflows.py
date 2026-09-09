"""Atomic end-to-end workflows used by the VocabCraft CLI."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import torch
from transformers import MT5EncoderModel

from vocabcraft.artifacts import atomic_output_directory, read_json, write_json
from vocabcraft.benchmarking import (
    benchmark_encoder_forward,
    benchmark_mapping,
    serialized_size,
    theoretical_vocabulary_bytes,
)
from vocabcraft.config import VocabCraftConfig
from vocabcraft.evaluation.coverage import evaluate_coverage
from vocabcraft.evaluation.datasets import stream_jsonl
from vocabcraft.evaluation.encoder_equivalence import compare_encoder_models
from vocabcraft.evaluation.external_hook import run_external_evaluation
from vocabcraft.evaluation.generation import compare_greedy_generation
from vocabcraft.evaluation.non_inferiority import paired_bootstrap_non_inferiority
from vocabcraft.evaluation.teacher_forcing import compare_teacher_forcing
from vocabcraft.evaluation.validation_gates import aggregate_validation_gates
from vocabcraft.exceptions import ArtifactError, ConfigurationError, ValidationFailure
from vocabcraft.fallback import guard_original_ids
from vocabcraft.hashing import sha256_file, sha256_json
from vocabcraft.inventory import build_inventory
from vocabcraft.manifests import write_selection_artifacts
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5_guarded_generation import (
    GuardedMT5Generator,
    load_guarded_mt5_model,
)
from vocabcraft.models.mt5_seq2seq import load_compact_mt5_seq2seq_model
from vocabcraft.models.registry import (
    SupportedAdapter,
    artifact_family,
    load_adapter,
    load_encoder_model,
)
from vocabcraft.packs import (
    build_cold_pack,
    merge_packs,
    reconstruct_full_artifact,
)
from vocabcraft.reporting import (
    reproducibility_manifest,
    risk_report,
    write_json_markdown_report,
)
from vocabcraft.selection import (
    SelectionResult,
    observe_calibration,
    observe_critical_terms,
    select_profile,
)
from vocabcraft.tokenizers.fingerprint import tokenizer_behavior_hash

ExecutionMode = Literal["encoder_exact", "seq2seq_compact", "seq2seq_guarded"]


def _configured_adapter(identifier: str, config: VocabCraftConfig) -> SupportedAdapter:
    adapter = load_adapter(
        identifier,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if config.runtime.dtype in {"float32", "fp32"} and any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in adapter.model.parameters()
    ):
        raise ConfigurationError(
            "runtime.dtype requests float32 but source weights have another dtype; "
            "use auto to preserve the checkpoint dtype (compaction does not cast weights)"
        )
    return adapter


def _selection(
    adapter: SupportedAdapter, config: VocabCraftConfig
) -> tuple[SelectionResult, IdMapping]:
    records = build_inventory(adapter.tokenizer)
    observe_calibration(records, adapter.tokenizer, config.profile.calibration_files)
    observe_critical_terms(records, adapter.tokenizer, config.profile.critical_terms_files)
    selection = select_profile(
        records,
        adapter.tokenizer,
        config.profile,
        adapter.model.config,
        getattr(adapter.model, "generation_config", None),
    )
    mapping = IdMapping.from_retained(list(selection.retained_ids), adapter.model_vocab_size)
    return selection, mapping


def analyze_vocab_to_directory(
    model_identifier: str, config: VocabCraftConfig, output: str | Path
) -> dict[str, Any]:
    """Inspect tokenizer inventory, select a profile, and write deterministic manifests."""

    adapter = _configured_adapter(model_identifier, config)
    selection, mapping = _selection(adapter, config)
    summary = {
        "profile_id": config.profile.id,
        "source_model": model_identifier,
        "original_tokenizer_vocabulary_size": len(adapter.tokenizer),
        "original_model_vocabulary_size": adapter.model_vocab_size,
        "retained_vocabulary_size": len(selection.retained_ids),
        "excluded_tokenizer_ids": len(selection.excluded_ids),
        "excluded_model_rows": adapter.model_vocab_size - len(selection.retained_ids),
        "model_state_sha256": adapter.model_state_hash(),
        "tokenizer_sha256": adapter.tokenizer_hash(),
        "tokenizer_behavior_sha256": tokenizer_behavior_hash(adapter.tokenizer),
        "model_family": adapter.model.config.model_type,
    }
    with atomic_output_directory(output) as staging:
        write_selection_artifacts(staging, selection, config.profile.id)
        write_json(staging / "mapping-report.json", mapping.to_dict())
        write_json(staging / "analysis.json", summary)
        risk = risk_report()
        write_json_markdown_report(staging, "risk-report", risk, "VocabCraft risk report")
        write_json(
            staging / "reproducibility-manifest.json",
            reproducibility_manifest(
                {
                    **summary,
                    "profile_config_sha256": sha256_file(config.source_path),
                }
            ),
        )
    return summary


def build_profile_to_directory(
    model_identifier: str,
    config: VocabCraftConfig,
    mode: ExecutionMode,
    output: str | Path,
) -> dict[str, Any]:
    """Build a selected compact profile using its inspected model architecture."""

    adapter = _configured_adapter(model_identifier, config)
    selection, mapping = _selection(adapter, config)
    if mode == "encoder_exact":
        return adapter.build_encoder_profile(
            mapping, Path(output), config.profile.id, selection=selection
        )
    if mode == "seq2seq_compact":
        return adapter.build_seq2seq_profile(
            mapping, Path(output), config.profile.id, selection=selection
        )
    return adapter.build_guarded_profile(
        mapping, Path(output), config.profile.id, selection=selection
    )


def _load_artifact_mapping(root: Path) -> tuple[dict[str, Any], IdMapping]:
    metadata_object = read_json(root / "vocabcraft-metadata.json")
    mapping_object = read_json(root / "mapping-report.json")
    if not isinstance(metadata_object, dict) or not isinstance(mapping_object, dict):
        raise ArtifactError("artifact metadata and mapping reports must be JSON objects")
    mapping = IdMapping.from_dict(mapping_object)
    if metadata_object.get("compact_vocabulary_size") != len(mapping.new_to_old):
        raise ArtifactError("metadata vocabulary size differs from ID mapping")
    if metadata_object.get("original_model_vocabulary_size") != mapping.original_vocab_size:
        raise ArtifactError("metadata original vocabulary size differs from ID mapping")
    return metadata_object, mapping


def _validate_source_hashes(adapter: SupportedAdapter, metadata: dict[str, Any]) -> None:
    if adapter.model.config.model_type != artifact_family(metadata):
        raise ValidationFailure("original model family differs from compact artifact")
    if adapter.tokenizer_hash() != metadata.get("tokenizer_sha256"):
        raise ValidationFailure("original tokenizer hash differs from compact artifact")
    if adapter.model_state_hash() != metadata.get("model_state_sha256"):
        raise ValidationFailure("original model state hash differs from compact artifact")
    expected_behavior = metadata.get("tokenizer_behavior_sha256")
    if expected_behavior and tokenizer_behavior_hash(adapter.tokenizer) != expected_behavior:
        raise ValidationFailure("original tokenizer behavior differs from compact artifact")
    if metadata.get("model_family") == "xlm-roberta" and (
        sha256_json(getattr(adapter, "sentence_embedding_config", None))
        != metadata.get("sentence_embedding_sha256")
    ):
        raise ValidationFailure("original sentence embedding configuration differs from artifact")


def _source_ids(adapter: SupportedAdapter, text: str) -> list[int]:
    tokenize = getattr(adapter, "tokenize_text", None)
    if callable(tokenize):
        return cast(list[int], tokenize(text))
    return cast(list[int], adapter.tokenizer.encode(text, add_special_tokens=True))


def _encoder_validation(
    adapter: SupportedAdapter,
    compact_model: Any,
    mapping: IdMapping,
    config: VocabCraftConfig,
    data_path: str | Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    records = iter(stream_jsonl(data_path))
    while group := list(islice(records, config.runtime.batch_size)):
        rows: list[list[int]] = []
        supported_items: list[dict[str, Any]] = []
        for record in group:
            source_ids = _source_ids(adapter, record.source)
            decision = guard_original_ids(
                source_ids,
                mapping,
                adapter.tokenizer,
                config.profile.id,
                config.fallback.policy,
                record_missing_pieces=config.fallback.record_missing_pieces,
            )
            item: dict[str, Any] = {
                "id": record.id,
                "metadata": record.metadata,
                "guard": decision.to_dict(),
                "comparison": None,
            }
            results.append(item)
            if decision.compact_ids is not None:
                rows.append(source_ids)
                supported_items.append(item)
        if not rows:
            continue
        pad_id = adapter.tokenizer.pad_token_id
        if not isinstance(pad_id, int):
            raise ValidationFailure("encoder batching requires an original padding token")
        width = max(map(len, rows))
        padding_left = getattr(adapter.tokenizer, "padding_side", "right") == "left"
        padded = []
        masks = []
        for row in rows:
            padding = [pad_id] * (width - len(row))
            mask_padding = [0] * len(padding)
            padded.append(padding + row if padding_left else row + padding)
            masks.append(
                mask_padding + [1] * len(row) if padding_left else [1] * len(row) + mask_padding
            )
        comparison = compare_encoder_models(
            adapter.model,
            compact_model,
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
            mapping,
            config.validation,
            include_hidden_states=True,
            include_sentence_embeddings=bool(getattr(adapter, "sentence_embedding_config", None)),
        )
        for item, row_comparison in zip(supported_items, comparison["per_example"], strict=True):
            item["comparison"] = row_comparison
    return results


def validate_to_directory(
    original_identifier: str,
    compact_artifact: str | Path,
    config: VocabCraftConfig,
    data_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate coverage and the mode-appropriate model behavior per example."""

    root = Path(compact_artifact)
    metadata, mapping = _load_artifact_mapping(root)
    adapter = _configured_adapter(original_identifier, config)
    _validate_source_hashes(adapter, metadata)
    coverage = evaluate_coverage(
        adapter.tokenizer,
        mapping,
        [data_path],
        profile_id=config.profile.id,
        fallback_policy=config.fallback.policy,
        encode_text=lambda text: _source_ids(adapter, text),
        record_missing_pieces=config.fallback.record_missing_pieces,
    )
    mode = metadata.get("vocabcraft_mode")
    encoder_results: list[dict[str, Any]] = []
    teacher_results: list[dict[str, Any]] = []
    generation_results: list[dict[str, Any]] = []
    if mode == "encoder_exact":
        compact_model = load_encoder_model(root, metadata)
        encoder_results = _encoder_validation(adapter, compact_model, mapping, config, data_path)
    elif mode == "seq2seq_compact":
        compact_seq2seq = load_compact_mt5_seq2seq_model(root / "model")
        for record in stream_jsonl(data_path):
            source_ids = adapter.tokenizer.encode(record.source, add_special_tokens=True)
            source_decision = guard_original_ids(
                source_ids,
                mapping,
                adapter.tokenizer,
                config.profile.id,
                config.fallback.policy,
                record_missing_pieces=config.fallback.record_missing_pieces,
            )
            if source_decision.compact_ids is None:
                generation_results.append(
                    {"id": record.id, "guard": source_decision.to_dict(), "comparison": None}
                )
                continue
            generation_results.append(
                {
                    "id": record.id,
                    "metadata": record.metadata,
                    "comparison": compare_greedy_generation(
                        adapter.model,
                        compact_seq2seq,
                        adapter.tokenizer,
                        source_ids,
                        mapping,
                        maximum_output_length=32,
                    ),
                }
            )
            if record.target is not None:
                target_ids = adapter.tokenizer.encode(record.target, add_special_tokens=True)
                target_decision = guard_original_ids(
                    target_ids,
                    mapping,
                    adapter.tokenizer,
                    config.profile.id,
                    config.fallback.policy,
                    record_missing_pieces=config.fallback.record_missing_pieces,
                )
                if target_decision.compact_ids is None:
                    teacher_results.append(
                        {"id": record.id, "guard": target_decision.to_dict(), "comparison": None}
                    )
                else:
                    teacher_results.append(
                        {
                            "id": record.id,
                            "metadata": record.metadata,
                            "comparison": compare_teacher_forcing(
                                adapter.model,
                                compact_seq2seq,
                                source_ids,
                                target_ids,
                                mapping,
                                config.validation.teacher_forcing_max_absolute_difference,
                            ),
                        }
                    )
    elif mode == "seq2seq_guarded":
        compact_guarded = load_guarded_mt5_model(root / "model", mapping.original_vocab_size)
        generator = GuardedMT5Generator(
            compact_guarded,
            adapter.tokenizer,
            mapping,
            config.profile.id,
            config.fallback.policy,
            full_model=adapter.model,
            original_decoder_start_token_id=int(metadata["original_decoder_start_token_id"]),
            original_eos_token_id=int(metadata["original_eos_token_id"]),
        )
        for record in stream_jsonl(data_path):
            guarded_result = generator.generate_greedy(record.source, 32)
            guarded_comparison: dict[str, Any] = {
                "guarded_result": guarded_result,
                "fallback_protected": bool(guarded_result["fallback_occurred"]),
            }
            if guarded_result["generated"]:
                source_ids = adapter.tokenizer.encode(record.source, add_special_tokens=True)
                with torch.no_grad():
                    full_ids = (
                        cast(Any, adapter.model)
                        .generate(
                            input_ids=torch.tensor([source_ids], dtype=torch.long),
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=32,
                        )[0]
                        .tolist()
                    )
                guarded_comparison["full_original_ids"] = full_ids
                guarded_comparison["exact_original_token_id_match"] = (
                    guarded_result["original_output_ids"] == full_ids
                )
            generation_results.append(
                {
                    "id": record.id,
                    "metadata": record.metadata,
                    "comparison": guarded_comparison,
                }
            )
    else:
        raise ArtifactError(f"unsupported VocabCraft artifact mode: {mode!r}")
    gates = aggregate_validation_gates(
        str(mode),
        config.validation,
        total_examples=coverage.total_examples,
        missing_critical_examples=coverage.missing_critical_examples,
        new_unk_count=coverage.new_unk_count,
        encoder_results=encoder_results,
        teacher_results=teacher_results,
        generation_results=generation_results,
    )
    passed = gates.passed
    external_result: dict[str, Any] | None = None
    if config.external_evaluation is not None:
        external = config.external_evaluation
        original_external = run_external_evaluation(
            external.command,
            model_path=original_identifier,
            metrics_file=external.metrics_file,
            timeout_seconds=external.timeout_seconds,
        )
        compact_external = run_external_evaluation(
            external.command,
            model_path=root,
            metrics_file=external.metrics_file,
            timeout_seconds=external.timeout_seconds,
        )
        original_metrics = original_external["metrics"]
        compact_metrics = compact_external["metrics"]
        if not isinstance(original_metrics, dict) or not isinstance(compact_metrics, dict):
            raise ValidationFailure("external evaluation metrics must be JSON objects")
        metric_name = external.non_inferiority.metric
        original_metric = original_metrics.get(metric_name)
        compact_metric = compact_metrics.get(metric_name)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (original_metric, compact_metric)
        ):
            raise ValidationFailure(f"external metrics are missing numeric {metric_name!r}")
        original_metric = cast(float, original_metric)
        compact_metric = cast(float, compact_metric)
        original_scores = original_metrics.get("per_example_scores")
        compact_scores = compact_metrics.get("per_example_scores")
        if (original_scores is not None or compact_scores is not None) and (
            not isinstance(original_scores, list)
            or not isinstance(compact_scores, list)
            or not original_scores
            or len(original_scores) != len(compact_scores)
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in original_scores + compact_scores
            )
        ):
            raise ValidationFailure("external per-example scores must be finite, paired lists")
        statistical_result: dict[str, Any] | None = None
        aggregate_passed = (
            float(original_metric) - float(compact_metric)
            <= external.non_inferiority.maximum_allowed_drop
        )
        if (
            isinstance(original_scores, list)
            and isinstance(compact_scores, list)
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in original_scores + compact_scores
            )
        ):
            statistical_result = paired_bootstrap_non_inferiority(
                [float(value) for value in original_scores],
                [float(value) for value in compact_scores],
                maximum_allowed_drop=external.non_inferiority.maximum_allowed_drop,
            )
        external_result = {
            "metric": metric_name,
            "original_metric": float(original_metric),
            "compact_metric": float(compact_metric),
            "maximum_allowed_drop": external.non_inferiority.maximum_allowed_drop,
            "aggregate_threshold_passed": aggregate_passed,
            "paired_bootstrap": statistical_result,
            "statistical_non_inferiority_established": (
                statistical_result["non_inferiority_passed"]
                if statistical_result is not None
                else False
            ),
            "aggregate_only_warning": (
                None
                if statistical_result is not None
                else "Aggregate metrics do not establish statistical non-inferiority."
            ),
            "original_run": original_external,
            "compact_run": compact_external,
        }
        passed = (
            passed
            and aggregate_passed
            and (statistical_result is None or bool(statistical_result["non_inferiority_passed"]))
        )
    validation = {
        "mode": mode,
        "model_family": artifact_family(metadata),
        "passed": passed,
        "coverage": coverage.to_dict(),
        "encoder_equivalence": encoder_results,
        "teacher_forcing": teacher_results,
        "generation": generation_results,
        "external_evaluation": external_result,
        "strict_failures": gates.strict_failures,
        "comparison_counts": gates.comparison_counts,
        "encoder_batch_size": config.runtime.batch_size,
    }
    with atomic_output_directory(output) as staging:
        write_json_markdown_report(
            staging, "validation", validation, "VocabCraft validation report"
        )
        write_json_markdown_report(
            staging,
            "risk-report",
            risk_report(validation),
            "VocabCraft risk report",
        )
        write_json(
            staging / "reproducibility-manifest.json",
            reproducibility_manifest(
                {
                    "original_model": original_identifier,
                    "compact_artifact": str(root.resolve()),
                    "data_sha256": sha256_file(data_path),
                    "profile_config_sha256": sha256_file(config.source_path),
                }
            ),
        )
    return validation


def _vocabulary_parameter_count(model: Any, vocabulary_tensor_names: Sequence[str]) -> int:
    storages: dict[int, torch.Tensor] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name in vocabulary_tensor_names:
            storages[parameter.untyped_storage().data_ptr()] = parameter
    return sum(value.numel() for value in storages.values())


def _source_encoder_model(adapter: SupportedAdapter) -> Any:
    if adapter.model.config.model_type == "xlm-roberta":
        return adapter.model
    model = MT5EncoderModel(copy.deepcopy(adapter.model.config))
    source_state = adapter.model.state_dict()
    model.load_state_dict(
        {name: source_state[name].detach().clone() for name in model.state_dict()},
        strict=True,
    )
    model.eval()
    return model


def benchmark_to_directory(
    original_identifier: str,
    compact_artifact: str | Path,
    data_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Measure CPU mapping/encoder latency and separate theoretical/serialized sizes."""

    root = Path(compact_artifact)
    metadata, mapping = _load_artifact_mapping(root)
    start = time.perf_counter()
    adapter = load_adapter(original_identifier, revision=metadata.get("source_revision"))
    original_load_seconds = time.perf_counter() - start
    _validate_source_hashes(adapter, metadata)
    mode = metadata.get("vocabcraft_mode")
    start = time.perf_counter()
    if mode == "encoder_exact":
        compact_model: Any = load_encoder_model(root, metadata)
    elif mode == "seq2seq_compact":
        compact_model = load_compact_mt5_seq2seq_model(root / "model")
    elif mode == "seq2seq_guarded":
        compact_model = load_guarded_mt5_model(root / "model", mapping.original_vocab_size)
    else:
        raise ArtifactError(f"unsupported benchmark artifact mode: {mode!r}")
    compact_load_seconds = time.perf_counter() - start
    original_benchmark_model: Any = (
        _source_encoder_model(adapter) if mode == "encoder_exact" else adapter.model
    )
    with TemporaryDirectory(prefix="vocabcraft-original-") as temporary:
        original_serialized = Path(temporary) / "model"
        original_benchmark_model.save_pretrained(original_serialized, safe_serialization=True)
        actual_original_serialized_bytes = serialized_size(original_serialized)
    covered_source: list[int] | None = None
    all_rows: list[list[int]] = []
    tokenization_times: list[float] = []
    for record in stream_jsonl(data_path):
        start = time.perf_counter()
        source_ids = _source_ids(adapter, record.source)
        tokenization_times.append((time.perf_counter() - start) * 1000.0)
        if all(token_id in mapping.old_to_new for token_id in source_ids):
            all_rows.append(source_ids)
            if covered_source is None:
                covered_source = source_ids
    if covered_source is None:
        raise ValidationFailure("benchmark data contains no fully covered source example")
    compact_source = mapping.map_original_ids(covered_source)
    original_ids = torch.tensor([covered_source], dtype=torch.long)
    compact_ids = torch.tensor([compact_source], dtype=torch.long)
    original_forward = benchmark_encoder_forward(
        original_benchmark_model, original_ids, torch.ones_like(original_ids)
    )
    compact_forward = benchmark_encoder_forward(
        compact_model, compact_ids, torch.ones_like(compact_ids)
    )
    coverage = evaluate_coverage(
        adapter.tokenizer,
        mapping,
        [data_path],
        profile_id=str(metadata.get("profile_id", "unknown")),
        fallback_policy="full_model",
        encode_text=lambda text: _source_ids(adapter, text),
    )
    original_vocab = adapter.model_vocab_size
    compact_vocab = len(mapping.new_to_old)
    embedding_dimension = adapter.model.get_input_embeddings().weight.shape[1]
    vocabulary_names = [str(item["name"]) for item in adapter.list_vocabulary_tensors()]
    original_vocabulary_parameters = _vocabulary_parameter_count(
        original_benchmark_model, vocabulary_names
    )
    compact_vocabulary_parameters = _vocabulary_parameter_count(compact_model, vocabulary_names)
    original_total_parameters = sum(
        parameter.numel() for parameter in original_benchmark_model.parameters()
    )
    compact_total_parameters = sum(parameter.numel() for parameter in compact_model.parameters())
    report = {
        "mode": mode,
        "original_vocabulary_size": original_vocab,
        "retained_vocabulary_size": compact_vocab,
        "excluded_vocabulary_size": original_vocab - compact_vocab,
        "retained_percentage": 100.0 * compact_vocab / original_vocab,
        "original_vocabulary_dependent_parameter_count": original_vocabulary_parameters,
        "compact_vocabulary_dependent_parameter_count": compact_vocabulary_parameters,
        "original_non_vocabulary_parameter_count": (
            original_total_parameters - original_vocabulary_parameters
        ),
        "compact_non_vocabulary_parameter_count": (
            compact_total_parameters - compact_vocabulary_parameters
        ),
        "original_total_parameter_count": original_total_parameters,
        "compact_total_parameter_count": compact_total_parameters,
        "theoretical_single_embedding_bytes_original": theoretical_vocabulary_bytes(
            original_vocab, embedding_dimension
        ),
        "theoretical_single_embedding_bytes_compact": theoretical_vocabulary_bytes(
            compact_vocab, embedding_dimension
        ),
        "actual_compact_artifact_bytes": serialized_size(root),
        "actual_compact_serialized_model_bytes": serialized_size(root / "model"),
        "actual_original_serialized_bytes": actual_original_serialized_bytes,
        "original_model_load_seconds": original_load_seconds,
        "compact_model_load_seconds": compact_load_seconds,
        "tokenization_time_ms_mean": sum(tokenization_times) / len(tokenization_times),
        "mapping": benchmark_mapping(mapping, all_rows),
        "original_encoder_forward": original_forward,
        "compact_encoder_forward": compact_forward,
        "resident_memory_note": (
            "Peak process RSS includes both loaded source and compact models; it is not an "
            "isolated active-model memory measurement."
        ),
        "fallback_rate": coverage.to_dict()["fallback_rate"],
        "energy_measured": False,
    }
    with atomic_output_directory(output) as staging:
        write_json_markdown_report(staging, "benchmark", report, "VocabCraft benchmark report")
    return report


def build_pack_to_directory(
    model_identifier: str,
    compact_artifact: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build a cold fallback pack for every row absent from a compact artifact."""

    root = Path(compact_artifact)
    metadata, mapping = _load_artifact_mapping(root)
    adapter = load_adapter(model_identifier, revision=metadata.get("source_revision"))
    _validate_source_hashes(adapter, metadata)
    excluded = sorted(set(range(mapping.original_vocab_size)) - set(mapping.new_to_old))
    mode = metadata.get("vocabcraft_mode")
    if mode == "encoder_exact":
        source_model = _source_encoder_model(adapter)
        vocabulary_state = source_model.state_dict()
    elif mode in {"seq2seq_compact", "seq2seq_guarded"}:
        vocabulary_state = adapter.model.state_dict()
    else:
        raise ArtifactError(f"unsupported artifact mode for cold pack: {mode!r}")
    return build_cold_pack(
        vocabulary_state,
        excluded,
        mapping.original_vocab_size,
        output,
        source_model_hash=adapter.model_state_hash(),
        tokenizer_hash=adapter.tokenizer_hash(),
        source_revision=getattr(adapter.model.config, "_commit_hash", None),
        profile_id=str(metadata.get("profile_id", "unknown")),
        vocabulary_tensor_names=[
            str(item["name"])
            for item in adapter.list_vocabulary_tensors()
            if item["name"] in vocabulary_state
        ],
    )


def merge_packs_to_directory(packs: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    """Merge compatible packs deterministically."""

    return merge_packs(packs, output)


def reconstruct_full_to_directory(
    compact_artifact: str | Path, pack: str | Path, output: str | Path
) -> dict[str, Any]:
    """Reconstruct and reload the original-vocabulary model artifact."""

    return reconstruct_full_artifact(compact_artifact, pack, output)
