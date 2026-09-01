"""SentencePiece metadata extraction without changing the model file."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SentencePieceMetadata:
    """Piece-type flags exposed by a SentencePiece processor."""

    piece_type: str
    is_control: bool
    is_unknown: bool
    is_unused: bool
    is_byte: bool


_BYTE_PIECE = re.compile(r"<0x[0-9A-Fa-f]{2}>")


def _predicate(processor: Any, method_name: str, token_id: int) -> bool:
    method = getattr(processor, method_name, None)
    return bool(method(token_id)) if callable(method) else False


def sentencepiece_piece_metadata(
    tokenizer: Any, token_id: int, piece: str
) -> SentencePieceMetadata:
    """Read piece flags when available and use explicit lexical fallback only for bytes."""

    processor = getattr(tokenizer, "sp_model", None)
    is_unknown = _predicate(processor, "is_unknown", token_id)
    is_control = _predicate(processor, "is_control", token_id)
    is_unused = _predicate(processor, "is_unused", token_id)
    is_byte = _predicate(processor, "is_byte", token_id) or bool(_BYTE_PIECE.fullmatch(piece))
    if is_unknown:
        piece_type = "unknown"
    elif is_control:
        piece_type = "control"
    elif is_unused:
        piece_type = "unused"
    elif is_byte:
        piece_type = "byte"
    else:
        piece_type = "normal"
    return SentencePieceMetadata(piece_type, is_control, is_unknown, is_unused, is_byte)
