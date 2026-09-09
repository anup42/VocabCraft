"""Offline equivalence and failure tests for sentence embedding vocabulary pruning."""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

from vocabcraft.artifacts import read_json, write_json
from vocabcraft.benchmarking import benchmark_encoder_forward
from vocabcraft.config import ValidationConfig, load_config
from vocabcraft.evaluation.encoder_equivalence import compare_encoder_models
from vocabcraft.exceptions import (
    ArtifactError,
    UnsupportedModeError,
    UnsupportedModelError,
    ValidationFailure,
)
from vocabcraft.mappings import IdMapping
from vocabcraft.models.xlm_roberta import (
    WORD_EMBEDDING_NAME,
    CompactXLMRobertaEncoder,
    XLMRobertaAdapter,
    pool_sentence_embeddings,
)
from vocabcraft.packs import build_cold_pack, load_pack, reconstruct_vocabulary_tensors
from vocabcraft.workflows import benchmark_to_directory, validate_to_directory


def _fixture(
    path: Path, *, pooler: bool = True, sentence: bool = True, dtype: torch.dtype = torch.float32
) -> XLMRobertaAdapter:
    words = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "hello",
        "world",
        "foreign",
        "phone",
        "text",
        "query",
        "<mask>",
        "unused",
    ]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, "<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
        model_max_length=30,
    )
    torch.manual_seed(22)
    model = XLMRobertaModel(
        XLMRobertaConfig(
            vocab_size=12,
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            max_position_embeddings=32,
            bos_token_id=0,
            pad_token_id=1,
            eos_token_id=2,
        ),
        add_pooling_layer=pooler,
    ).eval()
    model.to(dtype=dtype)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if sentence:
        (path / "1_Pooling").mkdir()
        write_json(
            path / "modules.json",
            [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ],
        )
        write_json(
            path / "1_Pooling" / "config.json",
            {
                "embedding_dimension": 12,
                "pooling_mode": "mean",
                "include_prompt": True,
            },
        )
        write_json(path / "sentence_bert_config.json", {"max_seq_length": 8})
        write_json(
            path / "config_sentence_transformers.json",
            {
                "prompts": {"query": "query "},
                "default_prompt_name": None,
            },
        )
    return XLMRobertaAdapter.from_pretrained(str(path))


def _mapping() -> IdMapping:
    return IdMapping.from_retained([0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 12)


@pytest.mark.parametrize("pooler", [False, True])
def test_every_tensor_and_padded_encoder_output_is_preserved(tmp_path: Path, pooler: bool) -> None:
    adapter = _fixture(tmp_path / "source", pooler=pooler)
    mapping = _mapping()
    artifact = tmp_path / "compact"
    metadata = adapter.build_encoder_profile(mapping, artifact, "tiny")
    assert metadata["reloaded_all_tensors_exact"]
    compact = CompactXLMRobertaEncoder.from_artifact(artifact)
    assert (compact.model.pooler is not None) == pooler
    assert compact.model.embeddings.padding_idx == 1
    assert compact.model.embeddings.position_embeddings.padding_idx == 1
    assert compact.model.embeddings.word_embeddings.padding_idx == 1
    assert list(item["name"] for item in adapter.list_vocabulary_tensors()) == [WORD_EMBEDDING_NAME]
    for name, original in adapter.model.state_dict().items():
        expected = original[list(mapping.new_to_old)] if name == WORD_EMBEDDING_NAME else original
        assert torch.equal(compact.model.state_dict()[name], expected)
    original_features = adapter.tokenizer(
        ["hello world", "phone"], padding=True, return_tensors="pt"
    )
    compact_features = dict(original_features)
    compact_features["input_ids"] = torch.tensor(
        [mapping.map_original_ids(row) for row in original_features["input_ids"].tolist()]
    )
    with torch.no_grad():
        expected_output = adapter.model(**original_features)
        actual_output = compact.model(**compact_features)
    torch.testing.assert_close(
        actual_output.last_hidden_state, expected_output.last_hidden_state, rtol=0, atol=0
    )


def test_mean_pooling_batch_mask_fallback_and_normalization(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source")
    adapter.build_encoder_profile(_mapping(), tmp_path / "compact")
    compact = CompactXLMRobertaEncoder.from_artifact(tmp_path / "compact")
    texts = ["hello world", "phone", "foreign"]
    vectors, decisions = compact.embed_batch(texts)
    features = adapter.tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        source = adapter.model(**features).last_hidden_state
        expected = pool_sentence_embeddings(source, features["attention_mask"])
    for index in (0, 1):
        torch.testing.assert_close(vectors[index], expected[index], rtol=1e-5, atol=1e-6)
    assert vectors[2] is None
    assert decisions[2]["fallback_required"]
    assert decisions[2]["missing_original_ids"] == [6]
    output, unsupported = compact.encode("foreign")
    assert output is None and unsupported["fallback_required"]
    normalized, _ = compact.embed("hello world", normalize_embeddings=True)
    assert normalized is not None
    torch.testing.assert_close(torch.linalg.vector_norm(normalized), torch.tensor(1.0))
    assert compact.embed_batch([]) == ([], [])


def test_prompts_truncation_and_lowercasing_match_saved_sentence_behavior(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source")
    config_file = tmp_path / "source" / "sentence_bert_config.json"
    write_json(config_file, {"max_seq_length": 8, "do_lower_case": True})
    write_json(
        tmp_path / "source" / "config_sentence_transformers.json",
        {
            "prompts": {"query": "QUERY "},
            "default_prompt_name": "query",
        },
    )
    adapter = XLMRobertaAdapter.from_pretrained(str(tmp_path / "source"))
    adapter.build_encoder_profile(_mapping(), tmp_path / "compact")
    compact = CompactXLMRobertaEncoder.from_artifact(tmp_path / "compact")
    text = "HELLO world phone text hello world foreign"
    expected_ids = adapter.tokenizer.encode("query " + text.lower(), truncation=True, max_length=8)
    assert adapter.tokenize_text(text) == expected_ids
    output, decision = compact.encode(text)
    assert output is not None
    assert decision["original_ids"] == expected_ids
    assert len(expected_ids) == 8
    assert not decision["fallback_required"]  # Excluded suffix is also truncated by the source.
    with torch.no_grad():
        expected = adapter.model(input_ids=torch.tensor([expected_ids])).last_hidden_state
    torch.testing.assert_close(output.last_hidden_state, expected, rtol=0, atol=0)
    override, override_decision = compact.embed("HELLO", prompt="")
    assert override is not None
    assert 9 not in override_decision["original_ids"]
    with pytest.raises(ValueError, match="unknown sentence prompt"):
        compact.embed("hello", prompt_name="absent")


def test_special_ids_and_unsupported_generation_fail_closed(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source")
    with pytest.raises(ValidationFailure, match="special IDs"):
        adapter.build_encoder_profile(
            IdMapping.from_retained(list(range(10)), 12), tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()
    for method in (adapter.build_seq2seq_profile, adapter.build_guarded_profile):
        with pytest.raises(UnsupportedModeError, match="encoder-only"):
            method(_mapping(), tmp_path / "bad")


@pytest.mark.parametrize("change", ["pooling", "module", "prompt_pooling", "path", "truncate_dim"])
def test_unsupported_sentence_pipeline_is_rejected(tmp_path: Path, change: str) -> None:
    source = tmp_path / "source"
    adapter = _fixture(source)
    if change == "pooling":
        write_json(
            source / "1_Pooling" / "config.json",
            {
                "embedding_dimension": 12,
                "pooling_mode": "max",
            },
        )
    elif change == "prompt_pooling":
        write_json(
            source / "1_Pooling" / "config.json",
            {
                "embedding_dimension": 12,
                "pooling_mode": "mean",
                "include_prompt": False,
            },
        )
    elif change == "truncate_dim":
        write_json(source / "config_sentence_transformers.json", {"truncate_dim": 6})
    else:
        modules = read_json(source / "modules.json")
        modules[1]["type" if change == "module" else "path"] = (
            "sentence_transformers.models.Dense" if change == "module" else "../../outside"
        )
        write_json(source / "modules.json", modules)
    with pytest.raises(UnsupportedModelError):
        XLMRobertaAdapter(adapter.model, adapter.tokenizer, str(source))


def test_artifact_rejects_changed_tokenizer_or_pooling(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source")
    adapter.build_encoder_profile(_mapping(), tmp_path / "compact")
    path = tmp_path / "compact" / "sentence-embedding-config.json"
    config = read_json(path)
    config["max_seq_length"] = 5
    write_json(path, config)
    with pytest.raises(ArtifactError, match="checksum"):
        CompactXLMRobertaEncoder.from_artifact(tmp_path / "compact")


def test_plain_xlmr_does_not_invent_sentence_pooling(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source", sentence=False)
    adapter.build_encoder_profile(_mapping(), tmp_path / "compact")
    compact = CompactXLMRobertaEncoder.from_artifact(tmp_path / "compact")
    output, decision = compact.encode("hello")
    assert output is not None and not decision["fallback_required"]
    with pytest.raises(UnsupportedModeError, match="saved mean-pooling"):
        compact.embed("hello")


def test_source_and_compact_bfloat16_rows_keep_their_dtype(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source", dtype=torch.bfloat16)
    assert adapter.model.embeddings.word_embeddings.weight.dtype == torch.bfloat16
    adapter.build_encoder_profile(_mapping(), tmp_path / "compact")
    compact = CompactXLMRobertaEncoder.from_artifact(tmp_path / "compact")
    assert compact.model.embeddings.word_embeddings.weight.dtype == torch.bfloat16
    assert torch.equal(
        compact.model.embeddings.word_embeddings.weight,
        adapter.model.embeddings.word_embeddings.weight[list(_mapping().new_to_old)],
    )


def test_declared_word_axis_avoids_hidden_and_position_dimension_collision(tmp_path: Path) -> None:
    source = {
        WORD_EMBEDDING_NAME: torch.arange(144, dtype=torch.float32).reshape(12, 12),
        "embeddings.position_embeddings.weight": torch.ones(12, 12),
    }
    build_cold_pack(
        source,
        [6, 11],
        12,
        tmp_path / "pack",
        source_model_hash="fixture",
        tokenizer_hash="fixture",
        source_revision=None,
        profile_id="fixture",
        vocabulary_tensor_names=[WORD_EMBEDDING_NAME],
    )
    pack = load_pack(tmp_path / "pack")
    assert list(pack.tensors) == [WORD_EMBEDDING_NAME]
    mapping = _mapping()
    compact_state = dict(source)
    compact_state[WORD_EMBEDDING_NAME] = source[WORD_EMBEDDING_NAME][list(mapping.new_to_old)]
    restored = reconstruct_vocabulary_tensors(compact_state, mapping, pack)
    assert all(torch.equal(restored[name], value) for name, value in source.items())


def test_public_equivalence_preserves_xlmr_input_embeddings_and_sentence_pooling(
    tmp_path: Path,
) -> None:
    adapter = _fixture(tmp_path / "source")
    mapping = _mapping()
    adapter.build_encoder_profile(mapping, tmp_path / "compact")
    compact = CompactXLMRobertaEncoder.from_artifact(tmp_path / "compact")
    features = adapter.tokenizer(["hello world", "phone"], padding=True, return_tensors="pt")
    report = compare_encoder_models(
        adapter.model,
        compact.model,
        features["input_ids"],
        features["attention_mask"],
        mapping,
        ValidationConfig(),
        include_hidden_states=True,
        include_sentence_embeddings=True,
    )
    assert report["passed"]
    assert report["embedding_layer"]["maximum_absolute_difference"] == 0
    assert len(report["intermediate_hidden_states"]) == 2
    assert len(report["per_example"]) == 2
    for example in report["per_example"]:
        assert example["passed"]
        assert example["attention_mask_preserved"]
        assert example["sentence_embedding"]["maximum_absolute_difference"] == 0
        assert example["normalized_sentence_embedding"]["maximum_absolute_difference"] == 0


def test_public_benchmark_runs_complete_xlmr_embedding_and_encoder(tmp_path: Path) -> None:
    adapter = _fixture(tmp_path / "source")
    features = adapter.tokenizer(["hello world", "phone"], padding=True, return_tensors="pt")
    report = benchmark_encoder_forward(
        adapter.model,
        features["input_ids"],
        features["attention_mask"],
        warmups=0,
        iterations=1,
    )
    assert report["iterations"] == 1
    assert report["latency_ms_mean"] >= 0
    assert report["peak_process_resident_bytes"] > 0


def test_xlmr_validation_and_benchmark_workflows_include_pooling_and_exact_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    adapter = _fixture(source)
    artifact = tmp_path / "compact"
    adapter.build_encoder_profile(_mapping(), artifact)
    config = load_config(Path(__file__).resolve().parents[2] / "configs/profiles/ste-en-hi.yaml")
    data = tmp_path / "examples.jsonl"
    data.write_text(
        '{"id":"short","text":"hello"}\n'
        '{"id":"long","text":"hello world phone"}\n'
        '{"id":"fallback","text":"foreign"}\n',
        encoding="utf-8",
    )
    report = validate_to_directory(str(source), artifact, config, data, tmp_path / "validation")
    assert report["passed"]
    assert report["model_family"] == "xlm-roberta"
    assert len(report["encoder_equivalence"]) == 3
    for example in report["encoder_equivalence"][:2]:
        assert example["comparison"]["passed"]
        assert example["comparison"]["sentence_embedding"]["maximum_absolute_difference"] == 0
    assert report["encoder_equivalence"][2]["guard"]["fallback_required"]
    benchmark = benchmark_to_directory(str(source), artifact, data, tmp_path / "benchmark")
    assert benchmark["original_vocabulary_dependent_parameter_count"] == 12 * 12
    assert benchmark["compact_vocabulary_dependent_parameter_count"] == 10 * 12
    assert (
        benchmark["original_non_vocabulary_parameter_count"]
        == (benchmark["compact_non_vocabulary_parameter_count"])
    )
    assert benchmark["fallback_rate"] == pytest.approx(1 / 3)
