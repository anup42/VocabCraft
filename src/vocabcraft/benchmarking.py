"""CPU-first size, memory, and latency benchmarking helpers."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from vocabcraft.mappings import IdMapping


def serialized_size(path: str | Path) -> int:
    """Sum materialized file bytes under a file or directory."""

    source = Path(path)
    if source.is_file():
        return source.stat().st_size
    return sum(item.stat().st_size for item in source.rglob("*") if item.is_file())


def theoretical_vocabulary_bytes(vocabulary_size: int, embedding_dimension: int) -> dict[str, int]:
    """Report theoretical single-matrix storage independently by floating dtype."""

    elements = vocabulary_size * embedding_dimension
    return {"fp32": elements * 4, "fp16": elements * 2, "bf16": elements * 2}


def benchmark_encoder_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    warmups: int = 2,
    iterations: int = 10,
) -> dict[str, Any]:
    """Measure CPU encoder forward latency and resident memory without energy claims."""

    if iterations < 1 or warmups < 0:
        raise ValueError("iterations must be positive and warmups non-negative")
    model.eval()
    process = psutil.Process()
    encoder = model.get_encoder() if callable(getattr(model, "get_encoder", None)) else model
    with torch.no_grad():
        for _ in range(warmups):
            encoder(input_ids=input_ids, attention_mask=attention_mask)
        latencies: list[float] = []
        peak_rss = process.memory_info().rss
        for _ in range(iterations):
            start = time.perf_counter()
            encoder(input_ids=input_ids, attention_mask=attention_mask)
            latencies.append((time.perf_counter() - start) * 1000.0)
            peak_rss = max(peak_rss, process.memory_info().rss)
    return {
        "device": "cpu",
        "iterations": iterations,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_min": min(latencies),
        "latency_ms_max": max(latencies),
        "peak_process_resident_bytes": peak_rss,
        "energy_measured": False,
    }


def benchmark_mapping(
    mapping: IdMapping, rows: list[list[int]], iterations: int = 100
) -> dict[str, Any]:
    """Measure ID guard/map overhead separately from tokenization and model execution."""

    if not rows:
        raise ValueError("mapping benchmark requires at least one row")
    latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        for row in rows:
            mapping.map_original_ids(row)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "iterations": iterations,
        "batch_rows": len(rows),
        "mapping_time_ms_mean": statistics.mean(latencies),
        "mapping_time_ms_median": statistics.median(latencies),
    }
