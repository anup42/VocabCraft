"""Model inspection reporting entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vocabcraft.artifacts import atomic_output_directory, write_json
from vocabcraft.models.mt5 import MT5Adapter


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
        f"- Shared is encoder embedding: {tying['shared_is_encoder_embedding']}",
        f"- Shared is decoder embedding: {tying['shared_is_decoder_embedding']}",
        f"- Shared storage equals output head: {tying['shared_output_same_storage']}",
        f"- Configuration requests tied output embeddings: {tying['config_tie_word_embeddings']}",
        "",
        "## Safety note",
        "",
        "Relationships above are based on object identity and storage pointers, not equal values.",
        "A tokenizer/model vocabulary-size mismatch is reported explicitly and is not silently",
        "normalized.",
    ]
    return "\n".join(lines) + "\n"


def inspect_mt5_to_directory(
    model_identifier: str,
    output: str | Path,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Load mT5, inspect actual runtime structure, and atomically write reports."""

    adapter = MT5Adapter.from_pretrained(
        model_identifier,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    report = adapter.inspect_model()
    with atomic_output_directory(output) as staging:
        write_json(staging / "inspection.json", report)
        (staging / "inspection.md").write_text(inspection_markdown(report), encoding="utf-8")
    return report
