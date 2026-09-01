from pathlib import Path

from vocabcraft.evaluation.datasets import stream_jsonl


def test_stream_jsonl_preserves_metadata(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"id":"x","text":"hello","language":"en","custom":{"source":"local"}}\n',
        encoding="utf-8",
    )
    records = list(stream_jsonl(path))
    assert records[0].source == "hello"
    assert records[0].metadata["custom"] == {"source": "local"}
