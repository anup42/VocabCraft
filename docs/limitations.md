# Current limitations

- The supported production-quality path is `encoder_exact` for mT5 structures
  recognized by the current adapter.
- `seq2seq_compact` is experimental. Successful input guarding cannot prevent a
  removed output token from changing generation.
- `seq2seq_guarded` supports deterministic greedy decoding only. Sampling and
  beam search are explicitly rejected. A tied full output projection makes the
  mode unsupported.
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

