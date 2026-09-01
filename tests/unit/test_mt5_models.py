from pathlib import Path

import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from vocabcraft.exceptions import UnsupportedModeError
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.models.mt5_guarded_generation import GuardedMT5Generator, load_guarded_mt5_model
from vocabcraft.models.mt5_seq2seq import load_compact_mt5_seq2seq_model


def _tiny_model() -> MT5ForConditionalGeneration:
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
        tie_word_embeddings=False,
    )
    config.tie_word_embeddings = False
    return MT5ForConditionalGeneration(config).eval()


def test_runtime_mt5_inspection_uses_actual_shapes(fake_tokenizer: object) -> None:
    model = _tiny_model()
    adapter = MT5Adapter(model, fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    report = adapter.inspect_model()
    assert report["model_class"] == "MT5ForConditionalGeneration"
    assert report["configuration_vocabulary_size"] == 11
    assert report["shared_embedding_shape"] == [11, 16]
    assert report["output_head_shape"] == [11, 16]
    assert report["weight_tying"]["shared_encoder_same_storage"] == (
        model.shared.weight.data_ptr() == model.encoder.embed_tokens.weight.data_ptr()
    )
    assert (
        report["weight_tying"]["shared_output_same_storage"]
        == report["weight_tying"]["config_tie_word_embeddings"]
    )


def test_build_encoder_copies_rows_and_reloads(fake_tokenizer: object, tmp_path: Path) -> None:
    adapter = MT5Adapter(_tiny_model(), fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 11)
    destination = tmp_path / "encoder"
    metadata = adapter.build_encoder_profile(mapping, destination, "test")
    assert metadata["reloaded_row_copy_exact"]
    assert (destination / "model" / "model.safetensors").is_file()
    assert (destination / "mapping-report.json").is_file()


def test_build_experimental_seq2seq_copies_rows_and_reloads(
    fake_tokenizer: object, tmp_path: Path
) -> None:
    adapter = MT5Adapter(_tiny_model(), fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 11)
    destination = tmp_path / "seq2seq"
    metadata = adapter.build_seq2seq_profile(mapping, destination, "test")
    assert metadata["experimental"]
    assert metadata["reloaded_row_copy_exact"]


def test_guarded_greedy_matches_full_model_when_every_output_id_is_retained(
    fake_tokenizer: object, tmp_path: Path
) -> None:
    original = _tiny_model()
    adapter = MT5Adapter(original, fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained(list(range(11)), 11)
    destination = tmp_path / "guarded"
    metadata = adapter.build_guarded_profile(mapping, destination, "test")
    assert metadata["full_original_output_projection"]
    compact = load_guarded_mt5_model(destination / "model", 11)
    generator = GuardedMT5Generator(
        compact,
        fake_tokenizer,  # type: ignore[arg-type]
        mapping,
        "test",
        full_model=original,
        original_decoder_start_token_id=0,
        original_eos_token_id=1,
    )
    result = generator.generate_greedy("hello", 4)
    source = torch.tensor([[3, 1]], dtype=torch.long)
    with torch.no_grad():
        expected = original.generate(
            input_ids=source,
            do_sample=False,
            num_beams=1,
            max_new_tokens=4,
        )[0].tolist()
    assert not result["fallback_occurred"]
    assert result["original_output_ids"] == expected


def test_guarded_mode_rejects_tied_output_head(fake_tokenizer: object, tmp_path: Path) -> None:
    config = MT5Config(
        vocab_size=11,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
    )
    tied = MT5ForConditionalGeneration(config)
    adapter = MT5Adapter(tied, fake_tokenizer, "tied-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained(list(range(11)), 11)
    with pytest.raises(UnsupportedModeError, match="tied"):
        adapter.build_guarded_profile(mapping, tmp_path / "guarded")


def test_seq2seq_preserves_shared_inputs_with_untied_output(
    fake_tokenizer: object, tmp_path: Path
) -> None:
    config = MT5Config(
        vocab_size=11,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
    )
    source = MT5ForConditionalGeneration(config)
    source.lm_head = torch.nn.Linear(16, 11, bias=False)
    adapter = MT5Adapter(source, fake_tokenizer, "mixed-tying-fixture")  # type: ignore[arg-type]
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 11)
    destination = tmp_path / "seq2seq-mixed-tying"
    adapter.build_seq2seq_profile(mapping, destination, "test")
    compact = load_compact_mt5_seq2seq_model(destination / "model")
    assert compact.shared.weight.data_ptr() == compact.encoder.embed_tokens.weight.data_ptr()
    assert compact.shared.weight.data_ptr() == compact.decoder.embed_tokens.weight.data_ptr()
    assert compact.shared.weight.data_ptr() != compact.lm_head.weight.data_ptr()
