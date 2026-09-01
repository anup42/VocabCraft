from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vocabcraft.config import VocabCraftConfig, load_config
from vocabcraft.inventory import build_inventory
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.selection import (
    SelectionResult,
    observe_calibration,
    observe_critical_terms,
    select_profile,
)


def _integration_enabled() -> bool:
    return os.environ.get("RUN_MT5_INTEGRATION") == "1"


@pytest.fixture(scope="session")
def mt5_adapter() -> MT5Adapter:
    if not _integration_enabled():
        pytest.skip("set RUN_MT5_INTEGRATION=1 to download and exercise google/mt5-small")
    return MT5Adapter.from_pretrained("google/mt5-small")


@pytest.fixture(scope="session")
def en_hi_config() -> VocabCraftConfig:
    return load_config(Path("configs/profiles/en-hi.yaml"))


@pytest.fixture(scope="session")
def en_hi_selection(
    mt5_adapter: MT5Adapter, en_hi_config: VocabCraftConfig
) -> tuple[SelectionResult, IdMapping]:
    records = build_inventory(mt5_adapter.tokenizer)
    observe_calibration(records, mt5_adapter.tokenizer, en_hi_config.profile.calibration_files)
    observe_critical_terms(
        records, mt5_adapter.tokenizer, en_hi_config.profile.critical_terms_files
    )
    selection = select_profile(
        records,
        mt5_adapter.tokenizer,
        en_hi_config.profile,
        mt5_adapter.model.config,
        mt5_adapter.model.generation_config,
    )
    mapping = IdMapping.from_retained(list(selection.retained_ids), mt5_adapter.model_vocab_size)
    return selection, mapping


@pytest.fixture(scope="session")
def encoder_artifact(
    mt5_adapter: MT5Adapter,
    en_hi_config: VocabCraftConfig,
    en_hi_selection: tuple[SelectionResult, IdMapping],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    selection, mapping = en_hi_selection
    destination = tmp_path_factory.mktemp("vocabcraft-integration") / "encoder"
    mt5_adapter.build_encoder_profile(
        mapping, destination, en_hi_config.profile.id, selection=selection
    )
    return destination


@pytest.fixture(scope="session")
def seq2seq_artifact(
    mt5_adapter: MT5Adapter,
    en_hi_config: VocabCraftConfig,
    en_hi_selection: tuple[SelectionResult, IdMapping],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    selection, mapping = en_hi_selection
    destination = tmp_path_factory.mktemp("vocabcraft-integration") / "seq2seq"
    mt5_adapter.build_seq2seq_profile(
        mapping, destination, en_hi_config.profile.id, selection=selection
    )
    return destination


@pytest.fixture(scope="session")
def first_sequence_record(en_hi_config: VocabCraftConfig) -> Any:
    from vocabcraft.evaluation.datasets import stream_jsonl

    for record in stream_jsonl(en_hi_config.profile.calibration_files[0]):
        if record.target is not None:
            return record
    raise AssertionError("English-Hindi calibration must contain a seq2seq record")
