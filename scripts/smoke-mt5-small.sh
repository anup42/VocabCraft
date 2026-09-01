#!/usr/bin/env bash
set -euo pipefail

vocabcraft inspect-model \
  --model google/mt5-small \
  --output artifacts/inspection

vocabcraft analyze-vocab \
  --model google/mt5-small \
  --profile configs/profiles/en-hi.yaml \
  --output artifacts/en-hi-analysis

vocabcraft build-profile \
  --model google/mt5-small \
  --profile configs/profiles/en-hi.yaml \
  --mode encoder_exact \
  --output artifacts/mt5-small-en-hi-encoder

vocabcraft validate \
  --original google/mt5-small \
  --compact artifacts/mt5-small-en-hi-encoder \
  --profile configs/profiles/en-hi.yaml \
  --data examples/smoke-en-hi.jsonl \
  --output artifacts/validation

vocabcraft benchmark \
  --original google/mt5-small \
  --compact artifacts/mt5-small-en-hi-encoder \
  --data examples/smoke-en-hi.jsonl \
  --output artifacts/benchmark

