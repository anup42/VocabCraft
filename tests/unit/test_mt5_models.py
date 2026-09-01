from pathlib import Path

import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter


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
    return MT5ForConditionalGeneration(config).eval()


def test_runtime_mt5_inspection_uses_actual_shapes(fake_tokenizer: object) -> None:
    adapter = MT5Adapter(_tiny_model(), fake_tokenizer, "local-fixture")  # type: ignore[arg-type]
    report = adapter.inspect_model()
    assert report["model_class"] == "MT5ForConditionalGeneration"
    assert report["configuration_vocabulary_size"] == 11
    assert report["shared_embedding_shape"] == [11, 16]
    assert report["output_head_shape"] == [11, 16]
    assert report["weight_tying"]["shared_encoder_same_storage"]
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
