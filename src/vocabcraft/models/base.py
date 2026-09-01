"""Extension contract for model-specific vocabulary compaction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from vocabcraft.mappings import IdMapping


class ModelAdapter(ABC):
    """Model-specific operations kept outside generic inventory and mapping code."""

    @abstractmethod
    def inspect_model(self) -> dict[str, Any]:
        """Inspect runtime structure, vocabulary tensors, and weight relationships."""

    @abstractmethod
    def list_vocabulary_tensors(self) -> list[dict[str, Any]]:
        """List state tensors with a model-vocabulary-sized dimension."""

    @abstractmethod
    def detect_weight_tying(self) -> dict[str, Any]:
        """Report identity and storage relationships without comparing values."""

    @abstractmethod
    def build_encoder_profile(self, mapping: IdMapping, output: Path) -> dict[str, Any]:
        """Build and reload an exact compact encoder artifact."""

    @abstractmethod
    def build_seq2seq_profile(self, mapping: IdMapping, output: Path) -> dict[str, Any]:
        """Build and reload an experimental compact sequence-to-sequence artifact."""

    @abstractmethod
    def validate_structure(self) -> None:
        """Fail when the loaded checkpoint structure is unsupported."""
