"""Typed per-token inventory construction."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from vocabcraft.tokenizers.base import OriginalTokenizer
from vocabcraft.tokenizers.sentencepiece import sentencepiece_piece_metadata
from vocabcraft.unicode_analysis import analyze_piece

_EXTRA_ID = re.compile(r"<extra_id_\d+>")


@dataclass
class TokenRecord:
    """All observed and derived facts for one original tokenizer ID."""

    original_id: int
    piece: str
    piece_type: str
    is_special: bool
    is_control: bool
    is_unknown: bool
    is_unused: bool
    is_byte: bool
    is_extra_id: bool
    unicode_scripts: set[str]
    unicode_categories: set[str]
    source_count: int = 0
    target_count: int = 0
    document_count: int = 0
    critical_count: int = 0
    observed_languages: set[str] = field(default_factory=set)
    observed_domains: set[str] = field(default_factory=set)
    selection_reasons: set[str] = field(default_factory=set)
    tier: str = "cold_fallback"
    risk_notes: list[str] = field(default_factory=list)
    has_emoji: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert sets to deterministically ordered JSON values."""

        result = asdict(self)
        for name in (
            "unicode_scripts",
            "unicode_categories",
            "observed_languages",
            "observed_domains",
            "selection_reasons",
        ):
            result[name] = sorted(result[name])
        return result


def build_inventory(tokenizer: OriginalTokenizer) -> list[TokenRecord]:
    """Create one inventory record for every original tokenizer ID."""

    special_ids = set(tokenizer.all_special_ids)
    records: list[TokenRecord] = []
    for token_id in range(len(tokenizer)):
        converted = tokenizer.convert_ids_to_tokens(token_id)
        piece = converted if isinstance(converted, str) else str(converted)
        metadata = sentencepiece_piece_metadata(tokenizer, token_id, piece)
        analysis = analyze_piece(piece)
        records.append(
            TokenRecord(
                original_id=token_id,
                piece=piece,
                piece_type=metadata.piece_type,
                is_special=token_id in special_ids,
                is_control=metadata.is_control,
                is_unknown=metadata.is_unknown or token_id == tokenizer.unk_token_id,
                is_unused=metadata.is_unused,
                is_byte=metadata.is_byte,
                is_extra_id=bool(_EXTRA_ID.fullmatch(piece)),
                unicode_scripts=set(analysis.unicode_scripts),
                unicode_categories=set(analysis.unicode_categories),
                has_emoji=analysis.has_emoji,
            )
        )
    return records
