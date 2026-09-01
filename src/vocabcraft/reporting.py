"""Validation, benchmark, risk, and reproducibility report writers."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from vocabcraft.artifacts import write_json

RISK_WARNINGS = [
    "Finite calibration data is not proof of complete language coverage.",
    "Script compatibility is not formal tokenizer reachability.",
    (
        "An unchanged tokenizer does not help when an embedding row is unavailable; "
        "missing-ID guarding is mandatory."
    ),
    "Compact output vocabularies can alter generation.",
    "Greedy guarded generation does not prove sampling or beam-search safety.",
    "Product deployment requires hidden task evaluation.",
    "The real product-wide language list has not been supplied.",
    "Runtime NPU language-pack loading is not included in version 1.",
]


def write_json_markdown_report(
    output: Path, stem: str, payload: dict[str, Any], title: str
) -> None:
    """Write machine-readable JSON and a readable recursive summary."""

    write_json(output / f"{stem}.json", payload)
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")
    lines.extend(["", "See the adjacent JSON report for complete per-example details."])
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def risk_report(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Separate structural guarantees, empirical results, and unresolved risks."""

    return {
        "structural_guarantees": [
            "The original tokenizer is retained unchanged.",
            "Compact IDs are contiguous and mapped bijectively from retained original IDs.",
            "Missing original IDs require an explicit fallback decision.",
        ],
        "exact_row_copy_guarantees": [
            "Retained embedding rows are copied exactly before quantization.",
            "Generated artifacts are reloaded and retained rows are checked again.",
        ],
        "encoder_equivalence_for_fully_covered_inputs": (
            validation.get("encoder_equivalence") if validation else "not measured"
        ),
        "corpus_based_empirical_coverage": (
            validation.get("coverage") if validation else "not measured"
        ),
        "experimental_generation_results": (
            validation.get("generation") if validation else "not measured"
        ),
        "operational_fallback_protection": (
            "Unsupported IDs never map to UNK, PAD, zero, or another existing row."
        ),
        "unsupported_claims_and_unresolved_risks": RISK_WARNINGS,
    }


def reproducibility_manifest(extra: dict[str, Any]) -> dict[str, Any]:
    """Record runtime and dependency versions alongside artifact-specific hashes."""

    dependencies = {}
    for package in (
        "numpy",
        "pyyaml",
        "regex",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependencies,
        **extra,
    }
