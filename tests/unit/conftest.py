from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest


class FakeSentencePiece:
    def is_unknown(self, token_id: int) -> bool:
        return token_id == 2

    def is_control(self, token_id: int) -> bool:
        return token_id in {0, 1}

    def is_unused(self, token_id: int) -> bool:
        return False

    def is_byte(self, token_id: int) -> bool:
        return False


class FakeTokenizer:
    pieces: ClassVar[list[str]] = [
        "<pad>",
        "</s>",
        "<unk>",
        "▁hello",
        "▁फोन",
        "!",
        "▁東京",
        "<extra_id_0>",
        "▁Ravi",
        "@",
        "▁दिल्लीMetro",
    ]

    def __init__(self) -> None:
        self.sp_model = FakeSentencePiece()

    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def all_special_ids(self) -> list[int]:
        return [0, 1, 2, 7]

    @property
    def all_special_tokens(self) -> list[str]:
        return ["<pad>", "</s>", "<unk>", "<extra_id_0>"]

    @property
    def additional_special_tokens(self) -> list[str]:
        return ["<extra_id_0>"]

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 2

    @property
    def eos_token_id(self) -> int:
        return 1

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self.pieces[ids]
        return [self.pieces[token_id] for token_id in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.pieces.index(tokens) if tokens in self.pieces else 2
        return [self.pieces.index(token) if token in self.pieces else 2 for token in tokens]

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word in {"hello", "Hello"}:
                ids.append(3)
            elif word in {"फोन", "phone"}:
                ids.append(4)
            elif word == "Ravi":
                ids.append(8)
            elif word == "दिल्लीMetro":
                ids.append(10)
            elif "東京" in word:
                ids.append(6)
            elif word == "@":
                ids.append(9)
            else:
                ids.append(2)
        if add_special_tokens:
            ids.append(1)
        return ids

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return " ".join(self.pieces[token_id] for token_id in token_ids)

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        texts: Sequence[str] = [text] if isinstance(text, str) else text
        return {"input_ids": [self.encode(item) for item in texts]}

    def save_pretrained(self, destination: str | Path) -> tuple[str]:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        vocabulary = path / "fake-vocabulary.txt"
        vocabulary.write_text("\n".join(self.pieces), encoding="utf-8")
        return (str(vocabulary),)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
