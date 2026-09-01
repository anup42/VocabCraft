"""Corpus coverage measurement with explicit missing-token accounting."""

from __future__ import annotations

from collections import Counter
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
) -> CoverageReport:
    """Stream calibration records and measure source/target guard outcomes."""

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
    fallback_examples = 0
    affected: list[dict[str, Any]] = []
    for path in data_paths:
        for record in stream_jsonl(path):
            total_examples += 1
            source_ids = tokenizer.encode(record.source, add_special_tokens=True)
            source_tokens += len(source_ids)
            original_unk_count += sum(token_id == tokenizer.unk_token_id for token_id in source_ids)
            source_decision = guard_original_ids(
                source_ids, mapping, tokenizer, profile_id, fallback_policy
            )
            source_missing_occurrences = [
                token_id
                for token_id in source_ids
                if token_id in source_decision.missing_original_ids
            ]
            covered_source_tokens += len(source_ids) - len(source_missing_occurrences)
            decisions = [source_decision]
            if record.target is not None:
                target_ids = tokenizer.encode(record.target, add_special_tokens=True)
                target_tokens += len(target_ids)
                original_unk_count += sum(
                    token_id == tokenizer.unk_token_id for token_id in target_ids
                )
                target_decision = guard_original_ids(
                    target_ids, mapping, tokenizer, profile_id, fallback_policy
                )
                target_missing_occurrences = [
                    token_id
                    for token_id in target_ids
                    if token_id in target_decision.missing_original_ids
                ]
                covered_target_tokens += len(target_ids) - len(target_missing_occurrences)
                decisions.append(target_decision)
            missing = [
                token_id for decision in decisions for token_id in decision.missing_original_ids
            ]
            if not missing:
                fully_covered_examples += 1
                continue
            fallback_examples += 1
            missing_counts.update(missing)
            for decision in decisions:
                missing_pieces.update(decision.missing_pieces)
            language = record.source_language or "unknown"
            domain = record.domain or "unknown"
            missing_by_language.setdefault(language, Counter()).update(missing)
            missing_by_domain.setdefault(domain, Counter()).update(missing)
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
        compact_path_unk_count=original_unk_count,
        new_unk_count=0,
        fallback_examples=fallback_examples,
        affected_examples_by_length=affected,
        most_common_missing_pieces=missing_pieces.most_common(20),
    )
