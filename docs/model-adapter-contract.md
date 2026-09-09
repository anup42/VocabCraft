# Model adapter contract

A model adapter owns tensor semantics. Generic selection and mapping code must
not guess tensor names, weight tying, or special-ID behavior.

An adapter must:

1. Validate the loaded model/tokenizer classes and reject unsupported variants.
2. Identify vocabulary-dependent tensor names and axes, including applicable
   biases. A hidden or position dimension equal to the vocabulary size does not
   make that dimension a vocabulary axis.
3. Inspect object identity and storage pointers; equal values do not establish a
   tie.
4. Collect tokenizer, model, generation, control, added, and reserved token IDs.
5. Build exact encoder rows, remap special IDs, preserve source floating-point
   dtype, save, reload, and validate every applicable tensor.
6. Compact every applicable seq2seq input/output structure while preserving
   supported ties and reporting generation as experimental.
7. Export/reconstruct exact cold rows with model/tokenizer compatibility hashes.
8. Fail explicitly for any unrecognized tensor axis or unsafe execution mode.

## Supported XLM-R encoder

`XLMRobertaAdapter` supports `XLMRobertaModel` in `encoder_exact` mode. It compacts
only `embeddings.word_embeddings.weight`, preserving all transformer, position,
token-type, layer-normalization, and optional dense-pooler tensors. A source
masked-language-model head is not supported. Both seq2seq modes are rejected.

The original padding ID must keep its numeric value in the compact mapping:
XLM-R uses that ID when generating position indices. Model, tokenizer, word
embedding, and position embedding padding indices must agree. Native model
reload must have no missing, unexpected, or mismatched tensors.

For local Sentence Transformers checkpoints, the supported pipeline is exactly
Transformer followed by attention-masked mean Pooling. The adapter records the
saved prompts, default prompt, lowercasing, sequence limit, and pooling settings.
Unsupported modules or pooling modes fail explicitly. `CompactXLMRobertaEncoder`
applies those settings before guarding input IDs and offers optional L2
normalization at inference. A plain XLM-R checkpoint exposes encoder outputs but
does not acquire an invented sentence-pooling pipeline.

New artifacts include a tokenizer behavior fingerprint covering segmentation,
added/special tokens, and relevant preprocessing defaults. Sentence-pipeline
configuration has a separate checksum. Model validation must check the saved
artifact through the same validated loader used by inference and compare the
source sentence configuration with its recorded checksum.

## Future mBERT adapter

Inspect WordPiece vocabulary artifacts, input embeddings, tied/untied MLM
decoder, decoder bias, and token-type/position behavior. Do not assume the MLM
decoder shares storage merely because values initially match.

## Another SentencePiece model

Provide exact piece metadata through the existing inventory interface. Preserve
the serialized model and normalization configuration. Test the whitespace marker,
control/user-defined/byte pieces, added tokens, and tokenizer/model size mismatch.

## Another encoder-decoder model

Inspect encoder and decoder embeddings independently, every output projection or
bias, weight ties, decoder start/EOS/forced/suppressed/bad-word IDs, and generation
configuration. Implement teacher-forced retained-logit comparison and retain all
per-example generation divergences.

Each new adapter requires offline fixture tests and real-checkpoint validation
when that checkpoint is available. Encoder tests include padded batches,
unsupported-token fallback, and sentence-vector comparisons when pooling exists.
