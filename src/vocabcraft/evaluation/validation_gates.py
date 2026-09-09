"""Fail-closed aggregation of the comparisons exposed by the validation CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vocabcraft.config import ValidationConfig


@dataclass(frozen=True)
class ValidationGateResult:
    """Aggregate result and explicit reasons why validation did not pass."""

    passed: bool
    strict_failures: dict[str, Any]
    comparison_counts: dict[str, int]


def _comparisons(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        comparison
        for result in results
        if isinstance(comparison := result.get("comparison"), Mapping)
    ]


def aggregate_validation_gates(
    mode: str,
    validation: ValidationConfig,
    *,
    total_examples: int,
    missing_critical_examples: Sequence[str],
    new_unk_count: int,
    encoder_results: Sequence[Mapping[str, Any]] = (),
    teacher_results: Sequence[Mapping[str, Any]] = (),
    generation_results: Sequence[Mapping[str, Any]] = (),
) -> ValidationGateResult:
    """Require actual mode comparisons and apply every configured behavioral gate.

    Missing-token rows with ``comparison: None`` remain reported fallback cases;
    they neither pass a comparison nor hide a failure in another row. Guarded
    generation promises preservation of greedy original IDs, so divergence or an
    unsupported result always fails, independently of the experimental compact
    generation opt-in threshold.
    """

    if total_examples < 0 or new_unk_count < 0:
        raise ValueError("coverage counts must be non-negative")
    encoders = _comparisons(encoder_results)
    teachers = _comparisons(teacher_results)
    generations = _comparisons(generation_results)
    guarded_successes = [
        comparison
        for comparison in generations
        if isinstance(result := comparison.get("guarded_result"), Mapping)
        and result.get("generated") is True
        and result.get("unsupported_profile") is not True
    ]
    counts = {
        "encoder": len(encoders),
        "teacher_forcing": len(teachers),
        "generation": len(guarded_successes) if mode == "seq2seq_guarded" else len(generations),
    }
    if mode == "seq2seq_guarded":
        fallback_count = sum(
            comparison["guarded_result"].get("fallback_occurred") is True
            for comparison in guarded_successes
        )
        counts["guarded_fallback_generation"] = fallback_count
        counts["guarded_compact_generation"] = len(guarded_successes) - fallback_count
    supported_mode = mode in {"encoder_exact", "seq2seq_compact", "seq2seq_guarded"}
    if mode == "encoder_exact":
        relevant_comparisons = len(encoders)
    elif mode == "seq2seq_compact":
        relevant_comparisons = len(teachers) + len(generations)
    elif mode == "seq2seq_guarded":
        relevant_comparisons = len(guarded_successes)
    else:
        relevant_comparisons = 0
    encoder_failed = any(comparison.get("passed") is not True for comparison in encoders)
    teacher_failed = any(comparison.get("passed") is not True for comparison in teachers)
    guarded_unsupported = mode == "seq2seq_guarded" and (
        len(guarded_successes) != len(generations)
    )
    if mode == "seq2seq_guarded":
        generation_failed = any(
            comparison.get("exact_original_token_id_match") is not True
            for comparison in guarded_successes
        )
    elif mode == "seq2seq_compact" and validation.generation_exact_match_required:
        generation_failed = not generations or any(
            comparison.get("exact_original_token_id_match") is not True
            or comparison.get("exact_decoded_text_match") is not True
            for comparison in generations
        )
    else:
        generation_failed = False
    critical_exceeded = (
        len(missing_critical_examples) > validation.maximum_missing_critical_examples
    )
    unk_exceeded = new_unk_count > validation.maximum_new_unk_count
    failures = {
        "missing_critical_examples": list(missing_critical_examples),
        "new_unk_count": new_unk_count,
        "missing_critical_threshold_exceeded": critical_exceeded,
        "new_unk_threshold_exceeded": unk_exceeded,
        "unsupported_mode": not supported_mode,
        "empty_validation_data": total_examples == 0,
        "no_mode_comparisons": relevant_comparisons == 0,
        "encoder_equivalence_failed": encoder_failed,
        "teacher_forcing_failed": teacher_failed,
        "generation_exact_match_failed": generation_failed,
        "guarded_generation_unsupported": guarded_unsupported,
    }
    passed = not any(
        (
            critical_exceeded,
            unk_exceeded,
            not supported_mode,
            total_examples == 0,
            relevant_comparisons == 0,
            encoder_failed,
            teacher_failed,
            generation_failed,
            guarded_unsupported,
        )
    )
    return ValidationGateResult(passed, failures, counts)
