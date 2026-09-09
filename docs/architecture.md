# VocabCraft architecture

VocabCraft separates model-independent safety logic from model-specific tensor
rewriting.

## Conservative request flow

1. The source preprocessing and unchanged original tokenizer produce original
   token IDs. Sentence encoders also apply saved prompts and truncation settings.
2. `ProfiledTokenizer` asks `IdMapping` whether every ID is retained.
3. A fully covered sequence is mapped to contiguous compact IDs without changing
   order or length.
4. `encoder_exact` runs copied embeddings and unchanged encoder blocks.
5. A missing ID produces a `GuardDecision` with IDs, reason, policy, and optional
   missing pieces.
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
- `models.registry`: dispatch by inspected model architecture and validated
  compact-encoder loading.
- `models.mt5`: runtime structure inspection and mT5 artifact builders.
- `models.mt5_encoder`, `mt5_seq2seq`, `mt5_guarded_generation`: mode-specific
  runtime behavior.
- `models.xlm_roberta`: exact XLM-R word-row compaction and saved Sentence
  Transformers mean pooling.
- `tokenizers.fingerprint`: tokenizer behavior fingerprints independent of
  checkpoint location.
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

Guarded mT5 generation retains the full original output projection. It calls
Transformers' public `generate` with the original generation configuration and
original token IDs, so forced/suppressed tokens and other supported greedy
processors retain their meaning. Decoder embedding inputs are guarded and mapped
at each lookup. A missing row triggers a complete restart on the supplied full
model or an explicit unsupported result. Temporary hooks are removed even when
generation fails, and guarded calls sharing a model are serialized.

## XLM-R sentence encoders

Only the input word table is pruned. Position embeddings, token-type embeddings,
transformer blocks, and the optional dense pooler remain unchanged. The original
padding ID is preserved to keep XLM-R's automatically generated positions exact.

For a supported local Transformer-plus-mean-Pooling pipeline,
`CompactXLMRobertaEncoder.embed_batch` tokenizes with the saved sentence settings,
guards each padded row, evaluates supported rows, and applies attention-masked
mean pooling. It returns vectors or fallback decisions in input order. Optional
L2 normalization is applied after pooling. Unsupported rows do not receive
placeholder embeddings.

## Validation and execution

Encoder validation uses `runtime.batch_size`, preserves padding masks, and
compares each covered example. Sentence models additionally compare mean-pooled
and normalized vectors. Empty data or zero applicable comparisons cannot pass;
teacher-forcing failures and required generation matches feed the aggregate
result and CLI exit status.

Configuration supports CPU execution. `runtime.dtype: auto` uses the adapter
loading policy (XLM-R preserves source dtype); explicit `float32`/`fp32` requires loaded float32 weights and
is checked rather than silently converting an exact artifact.

Local external evaluators can use `{model_path}` and `{metrics_file}` arguments.
The metrics placeholder receives a unique temporary file per invocation, and the
report embeds the resulting JSON. Legacy fixed output paths must change during
the invocation; an unchanged metrics file is rejected as stale.

## Artifact lifecycle

Every writer uses a sibling staging directory. On success it atomically renames
the staging directory; on failure it removes staging. Source model/tokenizer,
profile, data, pack, and tensor hashes prevent accidental cross-checkpoint use.
New artifacts also record tokenizer behavior and, for XLM-R sentence encoders,
the complete supported sentence configuration. Reload checks those records;
older artifacts without behavior fingerprints retain more limited provenance.
