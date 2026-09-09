"""Explicit architecture dispatch for supported model adapters and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig

from vocabcraft.exceptions import UnsupportedModelError
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.models.mt5_encoder import CompactMT5Encoder
from vocabcraft.models.xlm_roberta import CompactXLMRobertaEncoder, XLMRobertaAdapter

SupportedAdapter = MT5Adapter | XLMRobertaAdapter


def load_adapter(
    identifier: str, *, revision: str | None = None, trust_remote_code: bool = False
) -> SupportedAdapter:
    """Inspect config first; never coerce a different architecture into mT5."""

    config = AutoConfig.from_pretrained(
        identifier, revision=revision, trust_remote_code=trust_remote_code
    )
    if config.model_type == "mt5":
        return MT5Adapter.from_pretrained(
            identifier, revision=revision, trust_remote_code=trust_remote_code
        )
    if config.model_type == "xlm-roberta":
        return XLMRobertaAdapter.from_pretrained(
            identifier, revision=revision, trust_remote_code=trust_remote_code
        )
    raise UnsupportedModelError(f"unsupported model architecture: {config.model_type!r}")


def artifact_family(metadata: dict[str, Any]) -> str:
    """Legacy VocabCraft artifacts were all mT5."""

    family = str(metadata.get("model_family", "mt5"))
    if family not in {"mt5", "xlm-roberta"}:
        raise UnsupportedModelError(f"unsupported artifact model family: {family!r}")
    return family


def load_encoder_model(root: Path, metadata: dict[str, Any]) -> Any:
    """Reload the exact architecture represented by a compact encoder artifact."""

    if artifact_family(metadata) == "xlm-roberta":
        return CompactXLMRobertaEncoder.from_artifact(root).model
    return CompactMT5Encoder.from_artifact(root).model
