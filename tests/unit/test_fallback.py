import pytest

from vocabcraft.exceptions import MissingTokenError, UnsupportedModeError
from vocabcraft.fallback import FallbackExecutor, ProfiledTokenizer, guard_original_ids
from vocabcraft.mappings import IdMapping


def test_missing_id_triggers_fallback_and_never_becomes_unk(fake_tokenizer: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2, 3], 11)
    decision = guard_original_ids(  # type: ignore[arg-type]
        [3, 6, 1], mapping, fake_tokenizer, "test", "full_model"
    )
    assert decision.fallback_required
    assert decision.compact_ids is None
    assert decision.missing_original_ids == [6]
    assert decision.missing_pieces == ["▁東京"]


def test_profiled_tokenizer_round_trip(fake_tokenizer: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2, 3, 8], 11)
    tokenizer = ProfiledTokenizer(fake_tokenizer, mapping, "test", "reject")  # type: ignore[arg-type]
    decision = tokenizer.encode("hello Ravi")
    assert decision.compact_ids is not None
    assert tokenizer.decode(decision.compact_ids) == "▁hello ▁Ravi </s>"


def test_reject_fallback_raises_missing_token(fake_tokenizer: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2], 11)
    decision = guard_original_ids([6], mapping, fake_tokenizer, "test", "reject")  # type: ignore[arg-type]
    with pytest.raises(MissingTokenError):
        FallbackExecutor("reject").execute(decision)


def test_unavailable_full_model_fails_closed(fake_tokenizer: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2], 11)
    decision = guard_original_ids(  # type: ignore[arg-type]
        [6], mapping, fake_tokenizer, "test", "full_model"
    )
    with pytest.raises(UnsupportedModeError, match="unavailable"):
        FallbackExecutor("full_model").execute(decision)


def test_profiled_batch_preserves_padding_and_attention_mask(fake_tokenizer: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2, 3, 8], 11)
    tokenizer = ProfiledTokenizer(fake_tokenizer, mapping, "test", "full_model")  # type: ignore[arg-type]
    batch = tokenizer.batch_encode(["hello Ravi", "hello"], padding=True)
    assert batch.attention_mask == [[1, 1, 1], [1, 1, 0]]
    assert batch.decisions[0].compact_ids is not None
    assert batch.decisions[1].compact_ids is not None
    assert batch.decisions[1].original_ids[-1] == 0
