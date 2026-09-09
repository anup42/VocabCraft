from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vocabcraft.config import ValidationConfig, load_config
from vocabcraft.evaluation.encoder_equivalence import compare_tensors
from vocabcraft.evaluation.non_inferiority import paired_bootstrap_non_inferiority
from vocabcraft.exceptions import ConfigurationError, MappingError, ValidationFailure
from vocabcraft.fallback import ProfiledTokenizer
from vocabcraft.mappings import IdMapping
from vocabcraft.tokenizers.fingerprint import tokenizer_behavior_hash


@pytest.mark.parametrize(
    "change",
    [
        {"device": "cuda"},
        {"dtype": "float16"},
        {"batch_size": True},
    ],
)
def test_unsupported_runtime_config_fails_explicitly(tmp_path: Path, change: dict) -> None:
    payload = yaml.safe_load(Path("configs/profiles/en-hi.yaml").read_text(encoding="utf-8"))
    payload["runtime"].update(change)
    config = tmp_path / "profile.yaml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="runtime"):
        load_config(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("fallback", "fail_closed_when_unavailable", False),
        ("profile", "strict", "false"),
        ("validation", "encoder_max_absolute_difference", float("nan")),
        ("validation", "maximum_new_unk_count", -1),
    ],
)
def test_invalid_safety_settings_fail(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    payload = yaml.safe_load(Path("configs/profiles/en-hi.yaml").read_text(encoding="utf-8"))
    payload[section][key] = value
    config = tmp_path / "profile.yaml"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(config)


def test_serialized_mapping_order_cannot_be_silently_changed() -> None:
    with pytest.raises(MappingError, match="order"):
        IdMapping.from_dict({"original_vocab_size": 10, "new_to_old": [0, 5, 2]})
    payload = IdMapping.from_retained([0, 2, 5], 10).to_dict()
    payload["old_to_new"]["2"] = 2
    with pytest.raises(MappingError, match="disagrees"):
        IdMapping.from_dict(payload)


@pytest.mark.parametrize("bad_id", [True, 1.0, "1"])
def test_runtime_mappings_reject_values_aliasing_integer_ids(bad_id: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2], 3)
    for operation in (mapping.map_original_ids, mapping.map_compact_ids, mapping.map_labels):
        with pytest.raises(MappingError, match="integers"):
            operation([bad_id])


@pytest.mark.parametrize("ids", [[], [True], [1.0]])
def test_empty_or_noninteger_mapping_fails(ids: list) -> None:
    with pytest.raises(ValidationFailure):
        IdMapping.from_retained(ids, 10)


def test_guard_can_omit_sensitive_missing_pieces(fake_tokenizer: object) -> None:
    mapping = IdMapping.from_retained([0, 1, 2, 3], 11)
    tokenizer = ProfiledTokenizer(
        fake_tokenizer, mapping, "test", "reject", record_missing_pieces=False
    )
    result = tokenizer.encode("東京")
    assert result.fallback_required and result.missing_original_ids == [6]
    assert result.missing_pieces == []
    batch = tokenizer.batch_encode(["hello", "東京"])
    assert batch.decisions[1].missing_pieces == []


def test_tokenizer_fingerprint_includes_behavior_settings(fake_tokenizer: object) -> None:
    before = tokenizer_behavior_hash(fake_tokenizer)
    fake_tokenizer.padding_side = "left"
    assert tokenizer_behavior_hash(fake_tokenizer) != before


def test_zero_vectors_are_equal_but_empty_tensors_are_not_evidence() -> None:
    result = compare_tensors(torch.zeros(2, 4), torch.zeros(2, 4))
    assert result.sequence_cosine_similarity == 1.0
    assert result.minimum_per_token_cosine_similarity == 1.0
    with pytest.raises(ValueError, match="empty"):
        compare_tensors(torch.empty(0, 4), torch.empty(0, 4))


def test_bootstrap_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_non_inferiority([1.0], [float("nan")], maximum_allowed_drop=0.1)


def test_encoder_cosine_gate_checks_localized_errors() -> None:
    from vocabcraft.evaluation.encoder_equivalence import _passes

    original = torch.tensor([[100.0, 0.0], [0.0, 0.01]])
    compact = torch.tensor([[100.0, 0.0], [0.01, 0.0]])
    tolerances = ValidationConfig(
        encoder_max_absolute_difference=1.0,
        encoder_mean_absolute_difference=1.0,
    )
    metrics = compare_tensors(original, compact)
    assert metrics.sequence_cosine_similarity > tolerances.minimum_encoder_cosine_similarity
    assert not _passes(metrics, tolerances)


def test_explicit_runtime_dtype_cannot_silently_change_source_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vocabcraft import workflows

    config = load_config("configs/profiles/en-hi.yaml")
    model = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
    adapter = SimpleNamespace(model=model)
    monkeypatch.setattr(workflows, "load_adapter", lambda *args, **kwargs: adapter)
    fp32_config = replace(config, runtime=replace(config.runtime, dtype="float32"))
    with pytest.raises(ConfigurationError, match="source weights"):
        workflows._configured_adapter("fixture", fp32_config)
    auto_config = replace(config, runtime=replace(config.runtime, dtype="auto"))
    assert workflows._configured_adapter("fixture", auto_config) is adapter
    assert model.weight.dtype == torch.bfloat16
