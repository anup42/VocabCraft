# VocabCraft in plain English

VocabCraft keeps the original tokenizer.

The tokenizer first produces the same token numbers as the original model.
VocabCraft checks whether the smaller model contains an embedding for every
token number. When all embeddings exist, VocabCraft translates the original
token numbers into the smaller model's internal token numbers.

When an embedding is missing, VocabCraft uses the configured fallback. It does
not silently replace the token with UNK.

Encoder-only comparison is relatively straightforward because the same token
sequence and the same copied embeddings can be given to the unchanged encoder.
VocabCraft still measures the output rather than assuming it matches.

Text generation is harder because the model may want to generate a token that
was removed from the compact output vocabulary. Removing even one output row can
change scores and probabilities for other rows.

Therefore, compact generation is experimental and measured separately. The
guarded research mode keeps the full output projection only when that is safe for
the checkpoint, checks every selected output token, and falls back if its decoder
embedding is missing.

