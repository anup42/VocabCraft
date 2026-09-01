from vocabcraft.models.mt5 import MT5Adapter


def test_load_and_inspect_google_mt5_small(mt5_adapter: MT5Adapter) -> None:
    report = mt5_adapter.inspect_model()
    assert report["model_class"] == "MT5ForConditionalGeneration"
    assert report["configuration_vocabulary_size"] > 0
    assert report["tokenizer_vocabulary_size"] > 0
    assert report["embedding_dimension"] > 0
    assert report["vocabulary_sized_tensors"]
    assert report["sentencepiece_model_type"] is not None
