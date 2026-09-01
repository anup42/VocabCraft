"""Token inventory and selection report serialization."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vocabcraft.artifacts import write_json
from vocabcraft.selection import SelectionResult


def selection_summary(selection: SelectionResult, profile_id: str) -> dict[str, Any]:
    """Build count and reason summaries without claiming language completeness."""

    reasons: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    for record in selection.records:
        reasons.update(record.selection_reasons)
        tiers[record.tier] += 1
    original_size = len(selection.records)
    retained_size = len(selection.retained_ids)
    return {
        "profile_id": profile_id,
        "original_vocabulary_size": original_size,
        "retained_vocabulary_size": retained_size,
        "excluded_vocabulary_size": len(selection.excluded_ids),
        "retained_percentage": (100.0 * retained_size / original_size) if original_size else 0.0,
        "reason_counts": dict(sorted(reasons.items())),
        "tier_counts": dict(sorted(tiers.items())),
        "warning": "Finite calibration coverage is not proof of complete language coverage.",
    }


def write_selection_artifacts(
    destination: Path, selection: SelectionResult, profile_id: str
) -> None:
    """Write required machine-readable inventory and readable selection summary."""

    with gzip.open(destination / "token-manifest.jsonl.gz", "wt", encoding="utf-8") as handle:
        for record in selection.records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    write_json(destination / "retained-ids.json", list(selection.retained_ids))
    write_json(destination / "excluded-ids.json", list(selection.excluded_ids))
    summary = selection_summary(selection, profile_id)
    write_json(destination / "selection-summary.json", summary)
    lines = [
        "# VocabCraft selection summary",
        "",
        f"- Profile: `{profile_id}`",
        f"- Original vocabulary: {summary['original_vocabulary_size']}",
        f"- Retained vocabulary: {summary['retained_vocabulary_size']}",
        f"- Excluded vocabulary: {summary['excluded_vocabulary_size']}",
        f"- Retained percentage: {summary['retained_percentage']:.4f}%",
        "",
        "Finite calibration data is not proof of complete language coverage. Excluded ordinary",
        "tokens remain classified as `cold_fallback`, not certified removable.",
    ]
    (destination / "selection-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
