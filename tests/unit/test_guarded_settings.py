from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from vocabcraft.exceptions import ArtifactError, UnsupportedModeError
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.models.mt5_guarded_generation import GuardedMT5Generator, load_guarded_mt5_model


def _models(
    tokenizer: Any, retained: list[int] | None = None
) -> tuple[MT5ForConditionalGeneration, GuardedMT5Generator]:
    torch.manual_seed(2026)
    config = MT5Config(
        vocab_size=11,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    config.tie_word_embeddings = False
    original = MT5ForConditionalGeneration(config).eval()
    original.generation_config.suppress_tokens = [2]
    # Removing ID 2 shifts source, forced output and suppression IDs, so passing
    # compact IDs to generation processors cannot accidentally pass these tests.
    mapping = IdMapping.from_retained(retained or [0, 1, 3, 4, 5, 6, 7, 8, 9, 10], 11)
    compact = copy.deepcopy(original)
    for component, source in (
        (compact, original.shared),
        (compact.encoder, original.encoder.embed_tokens),
        (compact.decoder, original.decoder.embed_tokens),
    ):
        embedding = torch.nn.Embedding.from_pretrained(
            source.weight.detach()[list(mapping.new_to_old)].clone()
        )
        if component is compact:
            compact.shared = embedding
        else:
            component.set_input_embeddings(embedding)
    compact.config.vocab_size = len(mapping.new_to_old)
    return original, GuardedMT5Generator(
        compact, tokenizer, mapping, "test", full_model=original
    )


def _expected(
    model: MT5ForConditionalGeneration, tokenizer: Any, text: str, length: int
) -> list[int]:
    source = torch.tensor([tokenizer.encode(text)])
    with torch.no_grad():
        return model.generate(
            input_ids=source,
            attention_mask=torch.ones_like(source),
            do_sample=False,
            num_beams=1,
            max_new_tokens=length,
        )[0].tolist()


@pytest.mark.parametrize(
    "settings",
    [
        {"forced_bos_token_id": 8},
        {"forced_bos_token_id": 8, "forced_eos_token_id": 9},
        {"suppress_tokens": [2, 3, 6], "begin_suppress_tokens": [8]},
        {"repetition_penalty": 1.5, "no_repeat_ngram_size": 2},
        {"encoder_repetition_penalty": 1.2, "encoder_no_repeat_ngram_size": 1},
        {"bad_words_ids": [[8], [4, 5]], "min_new_tokens": 2},
        {"forced_bos_token_id": 8, "eos_token_id": [1, 8]},
        {"forced_bos_token_id": 8, "eos_token_id": 8, "decoder_start_token_id": 7},
        {"decoder_start_token_id": None, "bos_token_id": 7},
        {"forced_bos_token_id": 1, "eos_token_id": None},
    ],
)
def test_guarded_preserves_original_generation_processors(
    fake_tokenizer: Any, settings: dict[str, Any]
) -> None:
    original, generator = _models(fake_tokenizer)
    original.generation_config.update(**settings)
    expected = _expected(original, fake_tokenizer, "hello", 4)
    result = generator.generate_greedy("hello", 4)
    assert result["original_output_ids"] == expected
    assert not result["fallback_occurred"]
    assert not generator.compact_model.decoder.embed_tokens._forward_pre_hooks


def test_missing_generated_embedding_restarts_full_request_with_same_constraints(
    fake_tokenizer: Any,
) -> None:
    original, generator = _models(fake_tokenizer, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10])
    original.generation_config.forced_bos_token_id = 8
    original.generation_config.forced_eos_token_id = 9
    result = generator.generate_greedy("hello", 4)
    assert result["fallback_occurred"]
    assert result["first_missing_generated_id"] == 8
    assert result["original_output_ids"] == _expected(original, fake_tokenizer, "hello", 4)
    assert not generator.compact_model.decoder.embed_tokens._forward_pre_hooks


def test_missing_input_restarts_full_request_with_same_constraints(fake_tokenizer: Any) -> None:
    original, generator = _models(fake_tokenizer, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    original.generation_config.forced_bos_token_id = 8
    original.generation_config.forced_eos_token_id = 9
    result = generator.generate_greedy("東京", 4)
    assert result["fallback_occurred"]
    assert result["original_output_ids"] == _expected(original, fake_tokenizer, "東京", 4)


def test_missing_final_output_needs_no_decoder_embedding(fake_tokenizer: Any) -> None:
    original, generator = _models(fake_tokenizer, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10])
    original.generation_config.forced_eos_token_id = 8
    result = generator.generate_greedy("hello", 1)
    assert result["original_output_ids"] == [0, 8]
    assert not result["fallback_occurred"]


@pytest.mark.parametrize(
    "settings",
    [
        {"do_sample": True},
        {"num_beams": 2},
        {"num_return_sequences": 2},
        {"prompt_lookup_num_tokens": 2},
        {"stop_strings": ["done"]},
        {"max_time": 1},
        {"guidance_scale": 2},
        {"token_healing": True},
        {"cache_implementation": "paged"},
    ],
)
def test_guarded_rejects_unsupported_model_generation_settings(
    fake_tokenizer: Any, settings: dict[str, Any]
) -> None:
    original, generator = _models(fake_tokenizer)
    for name, value in settings.items():
        setattr(original.generation_config, name, value)
    with pytest.raises(UnsupportedModeError):
        generator.generate_greedy("hello", 4)
    assert not generator.compact_model.decoder.embed_tokens._forward_pre_hooks


@pytest.mark.parametrize("arguments", [{"do_sample": True}, {"num_beams": 2}])
def test_guarded_rejects_explicit_sampling_and_beams(
    fake_tokenizer: Any, arguments: dict[str, Any]
) -> None:
    _, generator = _models(fake_tokenizer)
    with pytest.raises(UnsupportedModeError):
        generator.generate_greedy("hello", 4, **arguments)


def test_hook_and_generation_configuration_are_restored_on_error(
    fake_tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, generator = _models(fake_tokenizer)
    saved = generator.compact_model.generation_config

    def fail(**kwargs: Any) -> None:
        raise RuntimeError("generation failure")

    monkeypatch.setattr(generator.compact_model, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failure"):
        generator.generate_greedy("hello", 4)
    assert generator.compact_model.generation_config is saved
    assert not generator.compact_model.decoder.embed_tokens._forward_pre_hooks


def test_direct_compact_model_requires_original_generation_settings(fake_tokenizer: Any) -> None:
    original, generator = _models(fake_tokenizer)
    with pytest.raises(ArtifactError, match="original_generation_config"):
        GuardedMT5Generator(generator.compact_model, fake_tokenizer, generator.mapping, "test")
    wrapper = GuardedMT5Generator(
        generator.compact_model,
        fake_tokenizer,
        generator.mapping,
        "test",
        original_generation_config=original.generation_config,
    )
    assert wrapper._model_lock is generator._model_lock
    assert wrapper.generate_greedy("hello", 4)["original_output_ids"] == _expected(
        original, fake_tokenizer, "hello", 4
    )


def test_guarded_artifact_restores_original_generation_settings(
    fake_tokenizer: Any, tmp_path: Path
) -> None:
    original, generator = _models(fake_tokenizer)
    original.generation_config.forced_bos_token_id = 8
    original.generation_config.forced_eos_token_id = 9
    adapter = MT5Adapter(original, fake_tokenizer, "local-fixture")
    destination = tmp_path / "guarded"
    adapter.build_guarded_profile(generator.mapping, destination, "test")
    restored = GuardedMT5Generator.from_artifact(destination, fake_tokenizer)
    assert restored.compact_model.generation_config.forced_bos_token_id == 8
    assert restored.compact_model.generation_config.forced_eos_token_id == 9
    result = restored.generate_greedy("hello", 4)
    assert result["original_output_ids"] == _expected(original, fake_tokenizer, "hello", 4)
    assert not result["fallback_occurred"]
    loaded = load_guarded_mt5_model(destination / "model", 11)
    assert loaded.generation_config.forced_bos_token_id == 8
    (destination / "source-generation-config.json").unlink()
    with pytest.raises(ArtifactError, match="source-generation-config"):
        load_guarded_mt5_model(destination / "model", 11)
