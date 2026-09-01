from pathlib import Path

from vocabcraft.hashing import hash_directory, sha256_bytes, sha256_file, sha256_json


def test_hashes_are_stable(tmp_path: Path) -> None:
    file_path = tmp_path / "value.txt"
    file_path.write_text("VocabCraft", encoding="utf-8")
    assert sha256_file(file_path) == sha256_bytes(b"VocabCraft")
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_directory_hash_includes_relative_names(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_text("same", encoding="utf-8")
    (second / "b").write_text("same", encoding="utf-8")
    assert hash_directory(first) != hash_directory(second)
