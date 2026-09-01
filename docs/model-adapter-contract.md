# Model adapter contract

A model adapter owns tensor semantics. Generic selection and mapping code must
not guess tensor names, weight tying, or special-ID behavior.

An adapter must:

1. Validate the loaded model/tokenizer classes and reject unsupported variants.
2. Enumerate every tensor with a vocabulary-sized dimension, including biases.
3. Inspect object identity and storage pointers; equal values do not establish a
   tie.
4. Collect tokenizer, model, generation, control, added, and reserved token IDs.
5. Build exact encoder rows, remap special IDs, save, reload, and validate.
6. Compact every applicable seq2seq input/output structure while preserving
   supported ties and reporting generation as experimental.
7. Export/reconstruct exact cold rows with model/tokenizer compatibility hashes.
8. Fail explicitly for any unrecognized tensor axis or unsafe execution mode.

## XLM-R

Implement an encoder-only adapter that inspects the input word embedding table,
special IDs, tokenizer post-processing, and masked-language-model output head if
that head is included. Reuse the generic guard and encoder equivalence suite.

## mBERT

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

Each new adapter requires offline fixture tests and gated real-checkpoint tests.

