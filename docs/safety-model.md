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

## Measured, not assumed

- Encoder final/intermediate hidden states and embedding outputs.
- Calibration source/target coverage and fallback rate.
- Retained teacher-forced logits and loss differences.
- Deterministic generated IDs, text, EOS, length, and first divergence.
- Serialized size, resident memory, mapping overhead, and CPU latency.

## Not proven

Finite corpus coverage does not prove that all English or Hindi inputs are
covered. Script compatibility does not prove tokenizer reachability. Compact
output vocabularies can change softmax probabilities and generation. Greedy
guarded generation does not establish safety for sampling or beam search.

Strict validation fails for a missing critical example, missing mandatory row,
new UNK event, failed encoder tolerance, or any silently executed unsupported
request. Default tolerances are configurable starting points, not universal
production thresholds.

