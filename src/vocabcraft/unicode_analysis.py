"""Unicode and SentencePiece-aware token analysis."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import regex

SENTENCEPIECE_WHITESPACE = "▁"

_SCRIPT_NAMES = (
    "Latin",
    "Devanagari",
    "Common",
    "Inherited",
    "Arabic",
    "Bengali",
    "Cyrillic",
    "Greek",
    "Gujarati",
    "Gurmukhi",
    "Han",
    "Hangul",
    "Hebrew",
    "Hiragana",
    "Kannada",
    "Katakana",
    "Malayalam",
    "Myanmar",
    "Oriya",
    "Sinhala",
    "Tamil",
    "Telugu",
    "Thai",
)
_SCRIPT_PATTERNS = {name: regex.compile(rf"\A\p{{Script={name}}}\Z") for name in _SCRIPT_NAMES}
_EMOJI_PATTERN = regex.compile(r"\p{Extended_Pictographic}")


@dataclass(frozen=True)
class UnicodeAnalysis:
    """Script/category facts for a visible tokenizer piece."""

    visible_text: str
    unicode_scripts: frozenset[str]
    unicode_categories: frozenset[str]
    has_emoji: bool
    has_sentencepiece_whitespace: bool


def visible_piece(piece: str) -> str:
    """Remove SentencePiece's whitespace marker only for Unicode analysis."""

    return piece.replace(SENTENCEPIECE_WHITESPACE, "")


def unicode_script(character: str) -> str:
    """Return the Unicode Script property for one character when recognized."""

    if len(character) != 1:
        raise ValueError("unicode_script expects exactly one character")
    for name, pattern in _SCRIPT_PATTERNS.items():
        if pattern.fullmatch(character):
            return name
    return "Unknown"


def analyze_piece(piece: str) -> UnicodeAnalysis:
    """Analyze a tokenizer piece without changing its stored representation."""

    visible = visible_piece(piece)
    scripts = frozenset(unicode_script(character) for character in visible)
    categories = frozenset(unicodedata.category(character) for character in visible)
    return UnicodeAnalysis(
        visible_text=visible,
        unicode_scripts=scripts,
        unicode_categories=categories,
        has_emoji=bool(_EMOJI_PATTERN.search(visible)),
        has_sentencepiece_whitespace=SENTENCEPIECE_WHITESPACE in piece,
    )


def is_url_email_building_piece(text: str) -> bool:
    """Identify conservative URL/email building blocks."""

    return bool(text) and all(character in ":/@._-?&=%+#~" for character in text)
