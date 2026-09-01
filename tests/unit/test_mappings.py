from typing import ClassVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vocabcraft.exceptions import MappingError, MissingTokenError
from vocabcraft.mappings import IdMapping, remap_token_id_fields


@given(st.sets(st.integers(min_value=0, max_value=100), min_size=1))
def test_mapping_bijection_and_contiguity(retained: set[int]) -> None:
    mapping = IdMapping.from_retained(list(retained), 101)
    originals = sorted(retained)
    compact = mapping.map_original_ids(originals)
    assert compact == list(range(len(retained)))
    assert mapping.map_compact_ids(compact) == originals


def test_mapping_rejects_absent_original_id() -> None:
    mapping = IdMapping.from_retained([0, 2, 4], 6)
    with pytest.raises(MissingTokenError) as failure:
        mapping.map_original_ids([0, 3])
    assert failure.value.missing_ids == [3]


def test_mapping_preserves_negative_100_labels() -> None:
    mapping = IdMapping.from_retained([1, 3, 5], 6)
    assert mapping.map_labels([1, -100, 5]) == [0, -100, 2]


def test_special_and_generation_configuration_remapping() -> None:
    mapping = IdMapping.from_retained([0, 2, 4, 7, 9], 10)

    class Config:
        pad_token_id = 0
        eos_token_id = 2
        decoder_start_token_id = 0
        forced_eos_token_id = 9
        suppress_tokens: ClassVar[list[int]] = [4, 7]
        bad_words_ids: ClassVar[list[list[int]]] = [[2, 4]]

    result = remap_token_id_fields(Config(), mapping)
    assert result["eos_token_id"] == 1
    assert result["suppress_tokens"] == [2, 3]
    assert result["bad_words_ids"] == [[1, 2]]


def test_invalid_compact_id_fails() -> None:
    mapping = IdMapping.from_retained([0], 2)
    with pytest.raises(MappingError, match="outside"):
        mapping.map_compact_ids([1])
