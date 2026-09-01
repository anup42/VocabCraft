# VocabCraft architecture

VocabCraft separates model-independent safety logic from model-specific tensor
rewriting.

## Conservative request flow

1. The unchanged original tokenizer produces original token IDs.
2. `ProfiledTokenizer` asks `IdMapping` whether every ID is retained.
3. A fully covered sequence is mapped to contiguous compact IDs without changing
   order or length.
4. `encoder_exact` runs copied embeddings and unchanged encoder blocks.
5. A missing ID produces a `GuardDecision` with IDs, pieces, reason, and policy.
6. The caller invokes the configured full-model/callback fallback or rejects.

Mapping is deliberately separate from tokenization. The framework cannot turn
a missing original ID into UNK because `IdMapping.map_original_ids` raises before
returning any partially mapped sequence.

## Layers

- `config`, `evaluation.datasets`: validated YAML and streaming JSONL.
- `unicode_analysis`, `inventory`, `selection`: tokenizer inventory and
  independent union-of-reasons selection.
- `mappings`, `fallback`: bijection, original-ID guard, decoding back-map, and
  fallback dispatch.
- `models.base`: adapter extension contract.
- `models.mt5`: runtime structure inspection and mT5 artifact builders.
- `models.mt5_encoder`, `mt5_seq2seq`, `mt5_guarded_generation`: mode-specific
  runtime behavior.
- `packs`: exact excluded rows, compatibility checks, deterministic merging, and
  reconstruction.
- `evaluation`: coverage, encoder equivalence, teacher forcing, generation,
  bootstrap, and local private-evaluation hook.
- `workflows`, `cli`: atomic end-to-end operations and reports.

## mT5 handling

The adapter does not hardcode mT5-small tensor shapes. It checks the loaded
class, configuration vocabulary size, shared/encoder/decoder/output modules,
every state tensor with a vocabulary-sized dimension, and both object/storage
relationships. Non-leading vocabulary dimensions fail as unsupported.

Tokenizer and model vocabulary sizes may differ. Guards operate on tokenizer
IDs; mapping and cold-pack coverage use the model vocabulary size so model-only
tail rows can still be reconstructed exactly.

## Artifact lifecycle

Every writer uses a sibling staging directory. On success it atomically renames
the staging directory; on failure it removes staging. Source model/tokenizer,
profile, data, pack, and tensor hashes prevent accidental cross-checkpoint use.

