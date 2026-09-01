from dataclasses import replace
from pathlib import Path

import pytest

from vocabcraft.config import ProfileConfig
from vocabcraft.exceptions import ValidationFailure
from vocabcraft.inventory import build_inventory
from vocabcraft.selection import observe_calibration, observe_critical_terms, select_profile


def _profile(**changes: object) -> ProfileConfig:
    profile = ProfileConfig(
        id="test",
        description="test",
        languages=["en", "hi"],
        unicode_scripts=["Latin", "Devanagari"],
    )
    return replace(profile, **changes)


def test_mandatory_extra_and_critical_tokens_are_retained(
    fake_tokenizer: object, tmp_path: Path
) -> None:
    records = build_inventory(fake_tokenizer)  # type: ignore[arg-type]
    terms = tmp_path / "terms.txt"
    terms.write_text("Ravi\n", encoding="utf-8")
    observe_critical_terms(records, fake_tokenizer, [terms])  # type: ignore[arg-type]
    result = select_profile(records, fake_tokenizer, _profile())  # type: ignore[arg-type]
    assert {0, 1, 2, 7, 8} <= set(result.retained_ids)
    assert "critical_term" in result.records[8].selection_reasons


def test_calibration_counts_source_and_target_independently(
    fake_tokenizer: object, tmp_path: Path
) -> None:
    records = build_inventory(fake_tokenizer)  # type: ignore[arg-type]
    data = tmp_path / "data.jsonl"
    data.write_text(
        '{"id":"seq","source":"hello Ravi","target":"फोन","source_language":"en",'
        '"target_language":"hi","domain":"assistant","critical":true}\n',
        encoding="utf-8",
    )
    observe_calibration(records, fake_tokenizer, [data])  # type: ignore[arg-type]
    assert records[3].source_count == 1
    assert records[4].target_count == 1
    assert records[8].critical_count == 1
    assert records[4].observed_languages == {"hi"}


def test_manual_removal_cannot_remove_mandatory_token(fake_tokenizer: object) -> None:
    records = build_inventory(fake_tokenizer)  # type: ignore[arg-type]
    profile = _profile(manual_remove_ids=[1])
    with pytest.raises(ValidationFailure, match="protected ID 1"):
        select_profile(records, fake_tokenizer, profile)  # type: ignore[arg-type]


def test_unselected_tokens_are_cold_not_certified(fake_tokenizer: object) -> None:
    records = build_inventory(fake_tokenizer)  # type: ignore[arg-type]
    result = select_profile(records, fake_tokenizer, _profile())  # type: ignore[arg-type]
    assert 6 in result.excluded_ids
    assert result.records[6].tier == "cold_fallback"
    assert "not certified removable" in result.records[6].risk_notes[0]
