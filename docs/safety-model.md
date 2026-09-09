# Safety model

VocabCraft is fail-closed around missing embeddings.

## Guaranteed by structure

- The original tokenizer and SentencePiece model are not rewritten.
- Retained IDs map bijectively to contiguous compact IDs.
- Covered sequences preserve original token order and sequence length.
- Retained rows are exact copies before quantization and are checked after
  artifact reload.
- Unsupported original IDs cannot produce a compact sequence.
- Cold-pack conflicts and source/tokenizer hash mismatches are hard failures.
- XLM-R pruning changes only the input word table and keeps its original padding
  ID, preserving position-index generation and all other model tensors.
- New encoder artifacts verify saved tokenizer behavior at load time. XLM-R
  sentence configurations are checksummed and compared with source settings.

## Measured, not assumed

- Encoder final/intermediate hidden states and embedding outputs.
- Padded-batch per-example outputs and, for supported sentence encoders,
  attention-masked mean vectors and their L2-normalized counterparts.
- Calibration source/target coverage and fallback rate.
- Retained teacher-forced logits and loss differences.
- Deterministic generated IDs, text, EOS, length, and first divergence.
- Serialized size, resident memory, mapping overhead, and CPU latency.

Coverage counts missing token occurrences separately for source and target and
attributes each side to its declared language. Existing original UNKs are
reported separately from new UNKs measured after mapping compact IDs back to
original IDs. Compact-path counts exclude sequences that require fallback;
those fallback sequences cannot hide a newly introduced UNK.

## Not proven

Finite corpus coverage does not prove that all English or Hindi inputs are
covered. Script compatibility does not prove tokenizer reachability. Compact
output vocabularies can change softmax probabilities and generation. Greedy
guarded generation does not establish safety for sampling or beam search.

Validation fails when critical-example or new-UNK counts exceed configured
thresholds, when encoder or teacher-forcing comparisons fail, or when no
applicable comparison ran. Empty datasets fail. Compact generation ID/text
matches are enforced when `generation_exact_match_required` is enabled. Guarded
generation always requires matching original output IDs; unsupported results
fail even if other examples succeed. Verified full-model fallback can count as
an operational generation comparison, with fallback and compact counts reported
separately.

Guarded generation applies supported Transformers greedy processors in original
ID space. It restarts from the original request if a decoder embedding is missing.
Unavailable fallback handlers remain fail-closed. Sampling, beam search, and
unsupported generation settings are rejected.

External evaluation must produce fresh JSON metrics. Prefer `{metrics_file}` to
isolate each invocation and preserve any previously declared metrics file on
failure. Numeric metrics and paired score lists must be finite. Default
tolerances are configurable starting points, not universal production thresholds.
