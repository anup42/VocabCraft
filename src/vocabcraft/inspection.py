"""Model inspection reporting entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vocabcraft.artifacts import atomic_output_directory, write_json
from vocabcraft.models.registry import load_adapter


def inspection_markdown(report: dict[str, Any]) -> str:
    """Render the most important inspection facts in readable Markdown."""

    tying = report["weight_tying"]
    lines = [
        "# VocabCraft model inspection",
        "",
        f"- Model: `{report['model_identifier']}`",
        f"- Resolved revision: `{report['resolved_revision']}`",
        f"- Model class: `{report['model_class']}`",
        f"- Tokenizer class: `{report['tokenizer_class']}`",
        f"- Tokenizer vocabulary: {report['tokenizer_vocabulary_size']}",
        f"- Model vocabulary: {report['configuration_vocabulary_size']}",
        f"- Shared embedding shape: `{report['shared_embedding_shape']}`",
        f"- Output head shape: `{report['output_head_shape']}`",
        f"- Total parameters: {report['total_parameter_count']}",
        f"- Vocabulary-dependent parameters: {report['vocabulary_dependent_parameter_count']}",
        f"- Model state SHA-256: `{report['model_state_sha256']}`",
        f"- Tokenizer SHA-256: `{report['tokenizer_sha256']}`",
        "",
        "## Weight relationships",
        "",
        *(f"- {key}: `{value}`" for key, value in tying.items() if key != "pointers"),
        "",
        "## Safety note",
        "",
        "Relationships above are based on object identity and storage pointers, not equal values.",
        "A tokenizer/model vocabulary-size mismatch is reported explicitly and is not silently",
        "normalized.",
    ]
    return "\n".join(lines) + "\n"


def inspect_model_to_directory(
    model_identifier: str,
    output: str | Path,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Dispatch a supported architecture and atomically write inspection reports."""

    adapter = load_adapter(
        model_identifier,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    report = adapter.inspect_model()
    with atomic_output_directory(output) as staging:
        write_json(staging / "inspection.json", report)
        (staging / "inspection.md").write_text(inspection_markdown(report), encoding="utf-8")
    return report


# Preserve the original public entry point for existing clients.
inspect_mt5_to_directory = inspect_model_to_directory
