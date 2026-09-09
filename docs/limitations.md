# Current limitations

- `encoder_exact` supports recognized mT5 encoders and XLM-R encoders. Equivalence
  applies to covered inputs under the inspected model and preprocessing settings;
  task accuracy still requires application evaluation.
- XLM-R support is encoder-only. For local Sentence Transformers checkpoints,
  exactly Transformer followed by attention-masked mean Pooling is supported.
  Other pooling modes, additional modules, MLM heads, and prompt-excluding pooling
  are rejected. Source prompt, lowercasing, and truncation settings are retained.
- A Hub XLM-R identifier loads an encoder; discovery of remote Sentence
  Transformers pipeline files is not implemented. Use the complete local
  checkpoint directory for sentence-pipeline preservation.
- `seq2seq_compact` is experimental. Successful input guarding cannot prevent a
  removed output token from changing generation.
- `seq2seq_guarded` supports deterministic greedy decoding only. Sampling and
  beam search are explicitly rejected, as are unsupported generation settings
  such as wall-clock stopping and custom generation. A tied full output projection
  or inconsistent configured tying makes the mode unsupported. Full-model
  fallback requires a caller-supplied model and repeats the entire request.
- Configured execution is CPU-only. `auto` uses the adapter loading policy
  (XLM-R preserves source dtype); explicit `float32`/`fp32` checks loaded weights. GPU/NPU placement
  and arbitrary dtype conversion are not configured execution paths. Encoder
  validation honors the configured batch size; generation remains per example.
- New tokenizer fingerprints cover behavior beyond vocabulary bytes. Legacy
  artifacts lacking these fields cannot provide the same provenance guarantees;
  rebuild them to obtain the stronger checks.
- English-Hindi calibration is deliberately small and repository-owned. It is not
  a benchmark or proof of product-wide language coverage.
- The final 24-language product list has not been supplied; no list is invented.
- Runtime NPU pack loading is out of scope. Packs merge into offline contiguous
  checkpoints.
- Pruning and quantization are not combined. Quantization metadata must be rebuilt
  and independently validated in a later milestone.
- CPU benchmarking does not measure energy. No energy-savings claim is made.
- Aggregate-only private metrics cannot establish statistical non-inferiority;
  paired per-example scores are needed for the bootstrap interval.
- External commands using a fixed metrics path rely on observed file changes for
  freshness. The `{metrics_file}` placeholder provides stronger per-run isolation.
