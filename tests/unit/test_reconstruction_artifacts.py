"""Offline round trips through complete compact, cold-pack and restored artifacts."""

from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoTokenizer,
    MT5Config,
    MT5EncoderModel,
    MT5ForConditionalGeneration,
    PreTrainedTokenizerFast,
)

from vocabcraft.artifacts import read_json, write_json
from vocabcraft.exceptions import ArtifactError
from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.models.mt5_seq2seq import load_compact_mt5_seq2seq_model
from vocabcraft.packs import (
    build_cold_pack,
    load_pack,
    merge_packs,
    reconstruct_full_artifact,
    reconstruct_vocabulary_tensors,
)


def _tokenizer() -> PreTrainedTokenizerFast:
    words = [
        "<pad>",
        "</s>",
        "<unk>",
        "hello",
        "phone",
        "!",
        "foreign",
        "<extra_id_0>",
        "name",
        "@",
        "city",
    ]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, "<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=["<extra_id_0>"],
        model_max_length=32,
    )


def _adapter(tying: str) -> MT5Adapter:
    torch.manual_seed(902)
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
    config.tie_word_embeddings = tying != "untied"
    model = MT5ForConditionalGeneration(config).eval()
    if tying == "mixed":
        # This is the important mT5 case: inputs share storage but the output
        # projection is distinct despite config.tie_word_embeddings being true.
        model.lm_head = torch.nn.Linear(16, 11, bias=False)
        assert model.config.tie_word_embeddings
        assert model.shared.weight.data_ptr() == model.encoder.embed_tokens.weight.data_ptr()
        assert model.shared.weight.data_ptr() == model.decoder.embed_tokens.weight.data_ptr()
        assert model.shared.weight.data_ptr() != model.lm_head.weight.data_ptr()
    model.generation_config.forced_bos_token_id = 8
    model.generation_config.forced_eos_token_id = 10
    model.generation_config.suppress_tokens = [9]
    model.generation_config.begin_suppress_tokens = [5]
    model.generation_config.bad_words_ids = [[4, 8]]
    return MT5Adapter(model, _tokenizer(), "offline-reconstruction")


@pytest.mark.parametrize(
    ("mode", "tying"),
    [
        ("encoder_exact", "untied"),
        ("encoder_exact", "tied"),
        ("encoder_exact", "mixed"),
        ("seq2seq_compact", "untied"),
        ("seq2seq_compact", "tied"),
        ("seq2seq_compact", "mixed"),
        ("seq2seq_guarded", "untied"),
    ],
)
def test_mt5_artifact_reconstruction_preserves_every_tensor_and_generation_settings(
    tmp_path: Path, mode: str, tying: str
) -> None:
    adapter = _adapter(tying)
    mapping = IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 11)
    compact_path = tmp_path / "compact"
    build = {
        "encoder_exact": adapter.build_encoder_profile,
        "seq2seq_compact": adapter.build_seq2seq_profile,
        "seq2seq_guarded": adapter.build_guarded_profile,
    }[mode]
    metadata = build(mapping, compact_path, "roundtrip")
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in adapter.model.state_dict().items()
        if mode != "encoder_exact" or name == "shared.weight" or name.startswith("encoder.")
    }
    pack_path = tmp_path / "cold"
    build_cold_pack(
        source_state,
        [6],
        11,
        pack_path,
        source_model_hash=metadata["model_state_sha256"],
        tokenizer_hash=metadata["tokenizer_sha256"],
        source_revision=None,
        profile_id="roundtrip",
    )
    restored_path = tmp_path / "restored"
    report = reconstruct_full_artifact(compact_path, pack_path, restored_path)
    assert report["reconstruction_complete"]
    assert report["restored_vocabulary_size"] == 11
    reloaded: Any
    if mode == "encoder_exact":
        reloaded = MT5EncoderModel.from_pretrained(restored_path / "model").eval()
    else:
        reloaded = load_compact_mt5_seq2seq_model(restored_path / "model")
    restored_state = reloaded.state_dict()
    assert set(restored_state) == set(source_state)
    for name, expected in source_state.items():
        assert torch.equal(restored_state[name], expected), name
    restored_tokenizer = AutoTokenizer.from_pretrained(restored_path / "tokenizer")
    assert restored_tokenizer.encode("hello phone") == adapter.tokenizer.encode("hello phone")
    if mode != "encoder_exact":
        # IDs above excluded ID 6 must be restored to original coordinates.
        for name in (
            "forced_bos_token_id",
            "forced_eos_token_id",
            "suppress_tokens",
            "begin_suppress_tokens",
            "bad_words_ids",
        ):
            assert getattr(reloaded.generation_config, name) == getattr(
                adapter.model.generation_config, name
            )
        if tying == "mixed":
            assert (
                reloaded.shared.weight.data_ptr() == reloaded.encoder.embed_tokens.weight.data_ptr()
            )
            assert (
                reloaded.shared.weight.data_ptr() == reloaded.decoder.embed_tokens.weight.data_ptr()
            )
            assert reloaded.shared.weight.data_ptr() != reloaded.lm_head.weight.data_ptr()


def _pack(tmp_path: Path, name: str, ids: list[int]) -> Path:
    destination = tmp_path / name
    build_cold_pack(
        {"embedding.weight": torch.arange(28, dtype=torch.float32).reshape(7, 4)},
        ids,
        7,
        destination,
        source_model_hash="model",
        tokenizer_hash="tokenizer",
        source_revision=None,
        profile_id="empty-pack-test",
    )
    return destination


def test_zero_exclusion_packs_merge_and_reconstruct_without_inventing_rows(tmp_path: Path) -> None:
    first = _pack(tmp_path, "empty-a", [])
    second = _pack(tmp_path, "empty-b", [])
    merged_path = tmp_path / "merged"
    merge_packs([first, second], merged_path)
    merged = load_pack(merged_path)
    assert merged.original_ids == ()
    assert merged.tensors["embedding.weight"].shape == (0, 4)
    assert merged.manifest["tensors"]["embedding.weight"]["pack_shape"] == [0, 4]
    original = {"embedding.weight": torch.arange(28, dtype=torch.float32).reshape(7, 4)}
    restored = reconstruct_vocabulary_tensors(
        original, IdMapping.from_retained(list(range(7)), 7), merged
    )
    assert torch.equal(restored["embedding.weight"], original["embedding.weight"])


def test_empty_and_nonempty_pack_merge_preserves_rows_and_metadata(tmp_path: Path) -> None:
    first = _pack(tmp_path, "empty", [])
    second = _pack(tmp_path, "nonempty", [2, 5])
    merged_path = tmp_path / "merged"
    merge_packs([first, second], merged_path)
    merged = load_pack(merged_path)
    expected = load_pack(second)
    assert merged.original_ids == (2, 5)
    assert torch.equal(merged.tensors["embedding.weight"], expected.tensors["embedding.weight"])
    assert merged.manifest["tensors"]["embedding.weight"]["pack_shape"] == [2, 4]


@pytest.mark.parametrize("ids", [[2, 2], [5, 2], [2, 7], [True, 5]])
def test_corrupted_pack_id_lists_are_rejected_even_when_manifest_agrees(
    tmp_path: Path, ids: list[int]
) -> None:
    path = _pack(tmp_path, "pack", [2, 5])
    write_json(path / "original-ids.json", ids)
    manifest = read_json(path / "manifest.json")
    manifest["original_ids"] = ids
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ArtifactError, match="sorted, unique"):
        load_pack(path)
