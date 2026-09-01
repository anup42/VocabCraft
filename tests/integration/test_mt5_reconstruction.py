import copy
from pathlib import Path

import torch
from transformers import MT5EncoderModel

from vocabcraft.mappings import IdMapping
from vocabcraft.models.mt5 import MT5Adapter
from vocabcraft.packs import build_cold_pack, load_pack, reconstruct_vocabulary_tensors
from vocabcraft.selection import SelectionResult


def test_cold_pack_reconstructs_original_encoder_vocabulary_tensors(
    mt5_adapter: MT5Adapter,
    en_hi_selection: tuple[SelectionResult, IdMapping],
    encoder_artifact: Path,
    tmp_path: Path,
) -> None:
    _, mapping = en_hi_selection
    source_encoder = MT5EncoderModel(copy.deepcopy(mt5_adapter.model.config))
    full_state = mt5_adapter.model.state_dict()
    source_encoder.load_state_dict(
        {name: full_state[name] for name in source_encoder.state_dict()}, strict=True
    )
    excluded = sorted(set(range(mapping.original_vocab_size)) - set(mapping.new_to_old))
    pack_path = tmp_path / "cold-pack"
    build_cold_pack(
        source_encoder.state_dict(),
        excluded,
        mapping.original_vocab_size,
        pack_path,
        source_model_hash=mt5_adapter.model_state_hash(),
        tokenizer_hash=mt5_adapter.tokenizer_hash(),
        source_revision=getattr(mt5_adapter.model.config, "_commit_hash", None),
        profile_id="en-hi-v1",
    )
    compact = MT5EncoderModel.from_pretrained(encoder_artifact / "model")
    reconstructed = reconstruct_vocabulary_tensors(
        compact.state_dict(), mapping, load_pack(pack_path)
    )
    original = source_encoder.state_dict()
    assert set(reconstructed) == set(original)
    assert all(torch.equal(reconstructed[name], original[name]) for name in original)
