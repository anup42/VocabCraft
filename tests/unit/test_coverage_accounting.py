import json
from pathlib import Path
from typing import Any

import pytest

from vocabcraft.evaluation.coverage import evaluate_coverage
from vocabcraft.mappings import IdMapping


def _dataset(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


def test_missing_counts_count_occurrences_and_attribute_target_language(
    fake_tokenizer: Any, tmp_path: Path
) -> None:
    data = _dataset(
        tmp_path / "data.jsonl",
        [
            {
                "id": "repeated",
                "source": "東京 東京 hello",
                "target": "फोन फोन फोन",
                "source_language": "ja",
                "target_language": "hi",
                "domain": "chat",
                "critical": True,
            }
        ],
    )
    report = evaluate_coverage(
        fake_tokenizer,
        IdMapping.from_retained([0, 1, 2, 3, 8], 11),
        [data],
        profile_id="test",
        fallback_policy="full_model",
    )
    assert report.missing_ids_by_count == {4: 3, 6: 2}
    assert report.missing_ids_by_language == {"hi": {4: 3}, "ja": {6: 2}}
    assert report.missing_ids_by_domain == {"chat": {4: 3, 6: 2}}
    assert report.most_common_missing_pieces == [("▁फोन", 3), ("▁東京", 2)]
    assert report.source_tokens == 4 and report.covered_source_tokens == 2
    assert report.target_tokens == 4 and report.covered_target_tokens == 1
    assert report.fallback_examples == 1
    assert report.missing_critical_examples == ["repeated"]


def test_missing_target_language_is_unknown_not_source_language(
    fake_tokenizer: Any, tmp_path: Path
) -> None:
    data = _dataset(
        tmp_path / "data.jsonl",
        [{"id": "target", "source": "hello", "target": "東京", "source_language": "en"}],
    )
    report = evaluate_coverage(
        fake_tokenizer,
        IdMapping.from_retained([0, 1, 2, 3], 11),
        [data],
        profile_id="test",
        fallback_policy="full_model",
    )
    assert report.missing_ids_by_language == {"unknown": {6: 1}}


def test_compact_unk_counts_exclude_fallback_and_retain_existing_unknowns(
    fake_tokenizer: Any, tmp_path: Path
) -> None:
    data = _dataset(
        tmp_path / "data.jsonl",
        [
            {"id": "covered", "text": "unknown hello"},
            {"id": "fallback", "text": "unknown unknown 東京"},
        ],
    )
    report = evaluate_coverage(
        fake_tokenizer,
        IdMapping.from_retained([0, 1, 2, 3], 11),
        [data],
        profile_id="test",
        fallback_policy="full_model",
    )
    assert report.original_unk_count == 3
    assert report.original_unk_count_on_compact_path == 1
    assert report.compact_path_unk_count == 1
    assert report.new_unk_count == 0
    assert report.fallback_examples == 1


def test_new_unknowns_are_measured_after_mapping_and_not_hidden_by_fallback(
    fake_tokenizer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset(
        tmp_path / "data.jsonl",
        [
            {"id": "covered", "text": "hello"},
            {"id": "fallback", "text": "unknown unknown 東京"},
        ],
    )
    # Reproduce a faulty compact roundtrip to make this a regression detector,
    # rather than assert a hardcoded zero on an untested compact mapping path.
    monkeypatch.setattr(IdMapping, "map_compact_ids", lambda self, ids: [2, 1])
    report = evaluate_coverage(
        fake_tokenizer,
        IdMapping.from_retained([0, 1, 2, 3], 11),
        [data],
        profile_id="test",
        fallback_policy="full_model",
    )
    assert report.original_unk_count == 2
    assert report.original_unk_count_on_compact_path == 0
    assert report.compact_path_unk_count == 1
    assert report.new_unk_count == 1


def test_encode_callback_applies_to_both_source_and_target_and_pieces_can_be_omitted(
    fake_tokenizer: Any, tmp_path: Path
) -> None:
    calls: list[str] = []

    def encode(text: str) -> list[int]:
        calls.append(text)
        return [3, 6, 6, 1]

    data = _dataset(
        tmp_path / "data.jsonl",
        [{"id": "callback", "source": "source", "target": "target"}],
    )
    report = evaluate_coverage(
        fake_tokenizer,
        IdMapping.from_retained([0, 1, 2, 3], 11),
        [data],
        profile_id="test",
        fallback_policy="full_model",
        encode_text=encode,
        record_missing_pieces=False,
    )
    assert calls == ["source", "target"]
    assert report.source_tokens == report.target_tokens == 4
    assert report.missing_ids_by_count == {6: 4}
    assert report.most_common_missing_pieces == []
