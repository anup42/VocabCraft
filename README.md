# VocabCraft

Safe, profile-aware vocabulary compaction for multilingual models.

VocabCraft builds smaller language-specific model profiles while preserving
tokenizer behavior, validating model quality, and explicitly handling missing
tokens. The first adapter supports `google/mt5-small`; the first demonstration
profile supports English, Hindi, Latin/Devanagari text, code switching,
romanized Hindi, names, product/location terms, URLs, email, numbers, symbols,
emoji, and common noisy input.

## Safety model

The conservative path keeps the original SentencePiece tokenizer unchanged. It
first obtains the original token IDs, checks that every ID has a retained row,
and only then maps the sequence to contiguous compact IDs. A missing ID is never
silently replaced by UNK, PAD, zero, a similar token, or an estimated embedding.
It produces an explicit full-model fallback, rejection, callback decision, or
strict failure.

`encoder_exact` preserves the covered token sequence, copied embedding rows,
attention mask, and encoder transformer weights. Numerical equivalence is still
measured for each covered example. `seq2seq_compact` is experimental because
removing output rows changes softmax normalization and can change generation.
`seq2seq_guarded` is greedy-only and is enabled only for checkpoints whose full
output projection is truly untied from the compact shared embedding; tied
checkpoints fail with an explicit unsupported-mode diagnostic.

See [safety-model.md](docs/safety-model.md) and
[limitations.md](docs/limitations.md) before deployment.

## Installation

Python 3.11 or newer and CPU execution are supported. CUDA is optional.

```powershell
python -m pip install -e ".[dev]"
vocabcraft --help
vocabcraft check-config --profile configs/profiles/en-hi.yaml
```

Model downloads are initiated only by model-dependent commands. Remote code is
disabled by default.

## mT5-small walkthrough

```powershell
vocabcraft inspect-model --model google/mt5-small --output artifacts/inspection

vocabcraft analyze-vocab `
  --model google/mt5-small `
  --profile configs/profiles/en-hi.yaml `
  --output artifacts/en-hi-analysis

vocabcraft build-profile `
  --model google/mt5-small `
  --profile configs/profiles/en-hi.yaml `
  --mode encoder_exact `
  --output artifacts/mt5-small-en-hi-encoder

vocabcraft validate `
  --original google/mt5-small `
  --compact artifacts/mt5-small-en-hi-encoder `
  --profile configs/profiles/en-hi.yaml `
  --data examples/smoke-en-hi.jsonl `
  --output artifacts/validation

vocabcraft benchmark `
  --original google/mt5-small `
  --compact artifacts/mt5-small-en-hi-encoder `
  --data examples/smoke-en-hi.jsonl `
  --output artifacts/benchmark

vocabcraft build-pack `
  --model google/mt5-small `
  --compact artifacts/mt5-small-en-hi-encoder `
  --output artifacts/cold-pack

vocabcraft reconstruct-full `
  --compact artifacts/mt5-small-en-hi-encoder `
  --pack artifacts/cold-pack `
  --output artifacts/reconstructed-encoder
```

The intentionally Japanese smoke example is not calibration data. It is used to
verify that an unsupported-script token triggers explicit fallback.

To study compact generation separately:

```powershell
vocabcraft build-profile `
  --model google/mt5-small `
  --profile configs/profiles/en-hi.yaml `
  --mode seq2seq_compact `
  --output artifacts/mt5-small-en-hi-seq2seq
```

Do not interpret a successful compact-generation run as a zero-risk or formal
equivalence result.

## Output artifacts

- `inspection.json` / `inspection.md`: actual model, tokenizer, tensor, special
  ID, hash, parameter, and storage-tie inspection.
- `token-manifest.jsonl.gz`: typed record and selection reasons for every
  tokenizer ID.
- `retained-ids.json`, `excluded-ids.json`, `mapping-report.json`: profile
  membership and bijective ID maps.
- `model/`: safetensors compact checkpoint; `tokenizer/` is an unchanged copy of
  the original tokenizer artifacts.
- `vocabcraft-metadata.json`, `source-config.json`: source hashes, sizes, mode,
  and exact reconstruction metadata.
- `validation.json` / `validation.md`: coverage, per-example equivalence,
  teacher-forced, generation, and strict-failure results.
- `benchmark.json` / `benchmark.md`: separately labeled theoretical bytes,
  serialized size, process memory, mapping overhead, and runtime latency.
- `risk-report.json` / `risk-report.md`: guarantees, empirical evidence, and
  unresolved risks.
- `reproducibility-manifest.json`: profile/data hashes and dependency versions.
- cold packs: `rows.safetensors`, original IDs, manifest, and checksums.

Artifact directories are staged beside their destination and published only
after successful completion. Existing directories are not overwritten.

## Tests and quality checks

Offline checks do not download a model:

```powershell
ruff check .
mypy src/vocabcraft
pytest tests/unit
```

Real-model integration is explicitly gated:

```powershell
$env:RUN_MT5_INTEGRATION = "1"
pytest tests/integration
```

Integration exercises real tokenizer metadata, inspection, selection, encoder
row copying/equivalence/fallback, experimental seq2seq logits/generation, cold
packs, and exact reconstruction.

## Extending VocabCraft

Generic inventory, Unicode analysis, mapping, guarding, packs, and evaluation
do not contain mT5-specific tensor logic. Add another adapter behind the
`ModelAdapter` contract, validate every vocabulary-sized tensor and special ID,
and add gated real-checkpoint tests. See
[model-adapter-contract.md](docs/model-adapter-contract.md).

## Current limitations

- Finite English-Hindi calibration is not proof of complete language coverage.
- Script compatibility is a conservative inclusion heuristic, not formal
  tokenizer reachability.
- Product deployment still requires hidden task evaluation.
- The final product-wide 24-language list has not been supplied; the example
  profile intentionally keeps that list empty.
- Dynamic live NPU pack loading and pruning-plus-quantization are not included.
- Guarded generation supports only deterministic greedy decoding with an
  explicit output limit; sampling and beam search are rejected.

