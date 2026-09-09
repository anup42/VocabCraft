"""Corpus coverage measurement with explicit missing-token accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vocabcraft.evaluation.datasets import stream_jsonl
from vocabcraft.fallback import FallbackPolicyName, guard_original_ids
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.base import OriginalTokenizer


@dataclass(frozen=True)
class CoverageReport:
    """Empirical source/target coverage without a language-safety claim."""

    total_examples: int
    fully_covered_examples: int
    source_tokens: int
    covered_source_tokens: int
    target_tokens: int
    covered_target_tokens: int
    missing_ids_by_count: dict[int, int]
    missing_ids_by_language: dict[str, dict[int, int]]
    missing_ids_by_domain: dict[str, dict[int, int]]
    missing_critical_examples: list[str]
    original_unk_count: int
    original_unk_count_on_compact_path: int
    compact_path_unk_count: int
    new_unk_count: int
    fallback_examples: int
    affected_examples_by_length: list[dict[str, Any]]
    most_common_missing_pieces: list[tuple[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible counts and percentages."""

        result = asdict(self)
        result.update(
            {
                "fully_covered_example_percentage": (
                    100.0 * self.fully_covered_examples / self.total_examples
                    if self.total_examples
                    else 0.0
                ),
                "source_token_coverage_percentage": (
                    100.0 * self.covered_source_tokens / self.source_tokens
                    if self.source_tokens
                    else 0.0
                ),
                "target_token_coverage_percentage": (
                    100.0 * self.covered_target_tokens / self.target_tokens
                    if self.target_tokens
                    else 0.0
                ),
                "fallback_rate": (
                    self.fallback_examples / self.total_examples if self.total_examples else 0.0
                ),
                "warning": (
                    "Finite calibration coverage is not proof of complete language coverage."
                ),
                "unk_accounting_note": (
                    "Original UNKs include every source/target sequence. Compact-path UNKs "
                    "are counted after original-to-compact-to-original mapping of fully "
                    "covered sequences only; fallback sequences are excluded."
                ),
            }
        )
        return result


def evaluate_coverage(
    tokenizer: OriginalTokenizer,
    mapping: IdMapping,
    data_paths: list[str | Path],
    *,
    profile_id: str,
    fallback_policy: FallbackPolicyName,
    encode_text: Callable[[str], list[int]] | None = None,
    record_missing_pieces: bool = True,
) -> CoverageReport:
    """Measure token occurrences and guards with optional model-specific preprocessing."""

    total_examples = 0
    fully_covered_examples = 0
    source_tokens = 0
    covered_source_tokens = 0
    target_tokens = 0
    covered_target_tokens = 0
    missing_counts: Counter[int] = Counter()
    missing_by_language: dict[str, Counter[int]] = {}
    missing_by_domain: dict[str, Counter[int]] = {}
    missing_pieces: Counter[str] = Counter()
    critical_missing: list[str] = []
    original_unk_count = 0
    original_unk_count_on_compact_path = 0
    compact_path_unk_count = 0
    new_unk_count = 0
    fallback_examples = 0
    affected: list[dict[str, Any]] = []
    for path in data_paths:
        for record in stream_jsonl(path):
            total_examples += 1
            source_ids = (
                encode_text(record.source)
                if encode_text is not None
                else tokenizer.encode(record.source, add_special_tokens=True)
            )
            source_tokens += len(source_ids)
            source_decision = guard_original_ids(
                source_ids, mapping, tokenizer, profile_id, fallback_policy,
                record_missing_pieces=record_missing_pieces,
            )
            source_missing = set(source_decision.missing_original_ids)
            source_missing_occurrences = [
                token_id for token_id in source_ids if token_id in source_missing
            ]
            covered_source_tokens += len(source_ids) - len(source_missing_occurrences)
            sequences = [(source_decision, source_missing_occurrences, record.source_language)]
            if record.target is not None:
                target_ids = (
                    encode_text(record.target)
                    if encode_text is not None
                    else tokenizer.encode(record.target, add_special_tokens=True)
                )
                target_tokens += len(target_ids)
                target_decision = guard_original_ids(
                    target_ids, mapping, tokenizer, profile_id, fallback_policy,
                    record_missing_pieces=record_missing_pieces,
                )
                target_missing = set(target_decision.missing_original_ids)
                target_missing_occurrences = [
                    token_id for token_id in target_ids if token_id in target_missing
                ]
                covered_target_tokens += len(target_ids) - len(target_missing_occurrences)
                sequences.append(
                    (target_decision, target_missing_occurrences, record.target_language)
                )
            missing: list[int] = []
            for decision, missing_occurrences, language in sequences:
                sequence_unk_count = sum(
                    token_id == tokenizer.unk_token_id for token_id in decision.original_ids
                )
                original_unk_count += sequence_unk_count
                if decision.compact_ids is not None:
                    roundtrip_ids = mapping.map_compact_ids(decision.compact_ids)
                    roundtrip_unk_count = sum(
                        token_id == tokenizer.unk_token_id for token_id in roundtrip_ids
                    )
                    original_unk_count_on_compact_path += sequence_unk_count
                    compact_path_unk_count += roundtrip_unk_count
                    new_unk_count += max(0, roundtrip_unk_count - sequence_unk_count)
                missing.extend(missing_occurrences)
                if not missing_occurrences:
                    continue
                missing_counts.update(missing_occurrences)
                missing_by_language.setdefault(language or "unknown", Counter()).update(
                    missing_occurrences
                )
                missing_by_domain.setdefault(record.domain or "unknown", Counter()).update(
                    missing_occurrences
                )
                if record_missing_pieces:
                    for token_id, count in Counter(missing_occurrences).items():
                        converted = tokenizer.convert_ids_to_tokens(token_id)
                        piece = converted if isinstance(converted, str) else str(converted)
                        missing_pieces[piece] += count
            if not missing:
                fully_covered_examples += 1
                continue
            fallback_examples += 1
            if record.critical:
                critical_missing.append(record.id)
            affected.append(
                {
                    "id": record.id,
                    "source_length": len(source_ids),
                    "missing_original_ids": sorted(set(missing)),
                    "metadata": record.metadata,
                }
            )
    affected.sort(key=lambda item: (-int(item["source_length"]), str(item["id"])))
    return CoverageReport(
        total_examples=total_examples,
        fully_covered_examples=fully_covered_examples,
        source_tokens=source_tokens,
        covered_source_tokens=covered_source_tokens,
        target_tokens=target_tokens,
        covered_target_tokens=covered_target_tokens,
        missing_ids_by_count=dict(sorted(missing_counts.items())),
        missing_ids_by_language={
            key: dict(sorted(value.items())) for key, value in sorted(missing_by_language.items())
        },
        missing_ids_by_domain={
            key: dict(sorted(value.items())) for key, value in sorted(missing_by_domain.items())
        },
        missing_critical_examples=critical_missing,
        original_unk_count=original_unk_count,
        original_unk_count_on_compact_path=original_unk_count_on_compact_path,
        compact_path_unk_count=compact_path_unk_count,
        new_unk_count=new_unk_count,
        fallback_examples=fallback_examples,
        affected_examples_by_length=affected,
        most_common_missing_pieces=missing_pieces.most_common(20),
    )
