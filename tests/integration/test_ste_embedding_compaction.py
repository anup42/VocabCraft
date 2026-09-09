"""Opt-in tests for an existing local STE XLM-R Sentence Transformer checkpoint.

Set RUN_STE_INTEGRATION=1 and STE_MODEL_PATH. STE_ARTIFACT_PATH optionally reuses
a completed profile; otherwise a single English/Hindi profile is built per session.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vocabcraft.artifacts import read_json
from vocabcraft.config import load_config
from vocabcraft.models.xlm_roberta import (
    CompactXLMRobertaEncoder,
    XLMRobertaAdapter,
    pool_sentence_embeddings,
)
from vocabcraft.packs import build_cold_pack, load_pack, reconstruct_vocabulary_tensors
from vocabcraft.workflows import _selection

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def ste_adapter() -> XLMRobertaAdapter:
    if os.environ.get("RUN_STE_INTEGRATION") != "1":
        pytest.skip("set RUN_STE_INTEGRATION=1 and STE_MODEL_PATH to test a local STE model")
    model_path = os.environ.get("STE_MODEL_PATH")
    if not model_path or not Path(model_path).is_dir():
        pytest.fail("STE_MODEL_PATH must name the existing local STE checkpoint directory")
    return XLMRobertaAdapter.from_pretrained(model_path)


@pytest.fixture(scope="session")
def ste_artifact(ste_adapter: XLMRobertaAdapter, tmp_path_factory: pytest.TempPathFactory) -> Path:
    existing = os.environ.get("STE_ARTIFACT_PATH")
    if existing:
        destination = Path(existing)
        metadata = read_json(destination / "vocabcraft-metadata.json")
        assert metadata["model_state_sha256"] == ste_adapter.model_state_hash()
        assert metadata["tokenizer_sha256"] == ste_adapter.tokenizer_hash()
        return destination
    config_path = Path(__file__).resolve().parents[2] / "configs/profiles/ste-en-hi.yaml"
    config = load_config(config_path)
    selection, mapping = _selection(ste_adapter, config)
    destination = tmp_path_factory.mktemp("vocabcraft-ste-integration") / "encoder"
    ste_adapter.build_encoder_profile(mapping, destination, config.profile.id, selection)
    return destination


@pytest.fixture(scope="session")
def ste_compact(ste_artifact: Path) -> CompactXLMRobertaEncoder:
    return CompactXLMRobertaEncoder.from_artifact(ste_artifact)


def test_real_ste_padded_token_states_and_sentence_embeddings(
    ste_adapter: XLMRobertaAdapter, ste_compact: CompactXLMRobertaEncoder
) -> None:
    texts = [
        "Where is my phone?",
        "मेरा फोन कहाँ है?",
        "Ravi, mera Galaxy phone कहाँ है?",
        "Visit https://example.in/a?q=फोन or email ravi@example.com.",
        "Meeting: 02/09/2026 at १०:३०; price ₹1,299.50!",
    ]
    original_rows = [ste_adapter.tokenize_text(text) for text in texts]
    width = max(map(len, original_rows))
    pad = ste_adapter.tokenizer.pad_token_id
    input_ids = torch.tensor([row + [pad] * (width - len(row)) for row in original_rows])
    attention_mask = torch.tensor(
        [[1] * len(row) + [0] * (width - len(row)) for row in original_rows]
    )
    mapped_ids = torch.tensor(
        [ste_compact.mapping.map_original_ids(row) for row in input_ids.tolist()]
    )
    assert ste_compact.mapping.old_to_new[pad] == pad
    assert ste_compact.model.embeddings.padding_idx == ste_adapter.model.embeddings.padding_idx
    assert ste_compact.model.embeddings.position_embeddings.padding_idx == pad
    with torch.no_grad():
        original = ste_adapter.model(input_ids=input_ids, attention_mask=attention_mask)
        compact = ste_compact.model(input_ids=mapped_ids, attention_mask=attention_mask)
    torch.testing.assert_close(
        compact.last_hidden_state, original.last_hidden_state, rtol=0, atol=1e-5
    )
    expected = pool_sentence_embeddings(original.last_hidden_state, attention_mask)
    embeddings, decisions = ste_compact.embed_batch(texts)
    assert all(not decision["fallback_required"] for decision in decisions)
    assert all(value is not None for value in embeddings)
    actual = torch.stack([value for value in embeddings if value is not None])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    normalized, _ = ste_compact.embed_batch(texts, normalize_embeddings=True)
    normalized_tensor = torch.stack([value for value in normalized if value is not None])
    torch.testing.assert_close(
        normalized_tensor,
        torch.nn.functional.normalize(expected, p=2, dim=1),
        rtol=1e-5,
        atol=1e-6,
    )
    assert actual.shape[1] == ste_adapter.model.config.hidden_size


def test_real_ste_unsupported_input_has_explicit_fallback(
    ste_compact: CompactXLMRobertaEncoder,
) -> None:
    texts = ["Where is my phone?", "東京へ行きます", "मेरा फोन कहाँ है?"]
    vectors, decisions = ste_compact.embed_batch(texts)
    assert vectors[0] is not None and vectors[2] is not None
    assert vectors[1] is None
    assert decisions[1]["fallback_required"]
    assert decisions[1]["compact_ids"] is None
    assert decisions[1]["missing_original_ids"]
    assert decisions[1]["missing_pieces"]


def test_real_ste_long_text_and_default_prompt_match_original_preprocessing(
    ste_adapter: XLMRobertaAdapter, ste_compact: CompactXLMRobertaEncoder
) -> None:
    text = "Where is my phone? " * 150
    original_ids = ste_adapter.tokenize_text(text)
    output, decision = ste_compact.encode(text)
    assert output is not None and decision["original_ids"] == original_ids
    assert ste_adapter.sentence_embedding_config is not None
    assert len(original_ids) <= ste_adapter.sentence_embedding_config["max_seq_length"]
    with torch.no_grad():
        expected = ste_adapter.model(input_ids=torch.tensor([original_ids])).last_hidden_state
    torch.testing.assert_close(output.last_hidden_state, expected, rtol=0, atol=1e-5)


def test_real_ste_cold_pack_reconstructs_every_original_tensor(
    ste_adapter: XLMRobertaAdapter, ste_compact: CompactXLMRobertaEncoder, tmp_path: Path
) -> None:
    mapping = ste_compact.mapping
    excluded = sorted(set(range(mapping.original_vocab_size)).difference(mapping.new_to_old))
    source = ste_adapter.model.state_dict()
    destination = tmp_path / "cold-pack"
    build_cold_pack(
        source,
        excluded,
        mapping.original_vocab_size,
        destination,
        source_model_hash=ste_adapter.model_state_hash(),
        tokenizer_hash=ste_adapter.tokenizer_hash(),
        source_revision=getattr(ste_adapter.model.config, "_commit_hash", None),
        profile_id=ste_compact.metadata["profile_id"],
        vocabulary_tensor_names=[item["name"] for item in ste_adapter.list_vocabulary_tensors()],
    )
    restored = reconstruct_vocabulary_tensors(
        ste_compact.model.state_dict(), mapping, load_pack(destination)
    )
    assert set(restored) == set(source)
    assert all(torch.equal(restored[name], tensor) for name, tensor in source.items())
