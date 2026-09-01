"""Deterministic original-ID and compact-ID mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vocabcraft.exceptions import MappingError, MissingTokenError, ValidationFailure


@dataclass(frozen=True)
class IdMapping:
    """Bijective mapping for retained IDs, ordered by original ID."""

    old_to_new: dict[int, int]
    new_to_old: tuple[int, ...]
    original_vocab_size: int

    @classmethod
    def from_retained(
        cls, retained_ids: list[int] | tuple[int, ...], original_vocab_size: int
    ) -> IdMapping:
        """Construct and validate a stable contiguous mapping."""

        ordered = tuple(sorted(retained_ids))
        if len(set(ordered)) != len(ordered):
            raise ValidationFailure("retained original IDs must be unique")
        if any(token_id < 0 or token_id >= original_vocab_size for token_id in ordered):
            raise ValidationFailure("retained original ID is outside the original vocabulary")
        old_to_new = {old_id: compact_id for compact_id, old_id in enumerate(ordered)}
        mapping = cls(old_to_new, ordered, original_vocab_size)
        mapping.validate()
        return mapping

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IdMapping:
        """Load a mapping from its machine-readable representation."""

        try:
            original_vocab_size = int(payload["original_vocab_size"])
            new_to_old = tuple(int(value) for value in payload["new_to_old"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MappingError(f"invalid mapping payload: {exc}") from exc
        return cls.from_retained(new_to_old, original_vocab_size)

    def validate(self) -> None:
        """Enforce contiguity, range, and round-trip invariants."""

        expected = set(range(len(self.new_to_old)))
        if set(self.old_to_new.values()) != expected:
            raise ValidationFailure("compact IDs must be contiguous from zero")
        if len(self.old_to_new) != len(self.new_to_old):
            raise ValidationFailure("mapping must be bijective")
        for compact_id, original_id in enumerate(self.new_to_old):
            if self.old_to_new.get(original_id) != compact_id:
                raise ValidationFailure("mapping round trip is inconsistent")
        if any(value < 0 or value >= self.original_vocab_size for value in self.new_to_old):
            raise ValidationFailure("original mapping ID is outside vocabulary")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible mapping representation."""

        return {
            "original_vocab_size": self.original_vocab_size,
            "compact_vocab_size": len(self.new_to_old),
            "old_to_new": {str(key): value for key, value in sorted(self.old_to_new.items())},
            "new_to_old": list(self.new_to_old),
        }

    def map_original_ids(self, original_ids: list[int]) -> list[int]:
        """Map only when every original ID exists; never coerce missing IDs."""

        missing = [token_id for token_id in original_ids if token_id not in self.old_to_new]
        if missing:
            raise MissingTokenError(missing)
        return [self.old_to_new[token_id] for token_id in original_ids]

    def map_labels(self, labels: list[int], ignore_index: int = -100) -> list[int]:
        """Map labels while preserving the framework ignore value exactly."""

        tokens = [label for label in labels if label != ignore_index]
        missing = [token_id for token_id in tokens if token_id not in self.old_to_new]
        if missing:
            raise MissingTokenError(missing)
        return [
            ignore_index if label == ignore_index else self.old_to_new[label] for label in labels
        ]

    def map_compact_ids(self, compact_ids: list[int]) -> list[int]:
        """Map generated compact IDs back to original IDs for decoding."""

        if any(token_id < 0 or token_id >= len(self.new_to_old) for token_id in compact_ids):
            raise MappingError("compact token ID is outside the compact vocabulary")
        return [self.new_to_old[token_id] for token_id in compact_ids]


def remap_token_id_fields(owner: Any, mapping: IdMapping) -> dict[str, Any]:
    """Return remapped special/generation fields without mutating the source object."""

    result: dict[str, Any] = {}
    scalar_fields = (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "decoder_start_token_id",
        "unk_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
    )
    list_fields = ("suppress_tokens", "begin_suppress_tokens")
    for name in scalar_fields:
        value = getattr(owner, name, None)
        if value is not None:
            if not isinstance(value, int):
                raise MappingError(f"{name} must be an integer or null")
            result[name] = mapping.map_original_ids([value])[0]
    for name in list_fields:
        value = getattr(owner, name, None)
        if value is not None:
            if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
                raise MappingError(f"{name} must be a list of integers or null")
            result[name] = mapping.map_original_ids(value)
    bad_words = getattr(owner, "bad_words_ids", None)
    if bad_words is not None:
        if not isinstance(bad_words, list) or not all(
            isinstance(sequence, list) and all(isinstance(item, int) for item in sequence)
            for sequence in bad_words
        ):
            raise MappingError("bad_words_ids must be a list of integer lists or null")
        result["bad_words_ids"] = [mapping.map_original_ids(sequence) for sequence in bad_words]
    return result
