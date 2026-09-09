"""Paired bootstrap support for private per-example evaluation scores."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_bootstrap_non_inferiority(
    original_scores: list[float],
    compact_scores: list[float],
    *,
    maximum_allowed_drop: float,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, Any]:
    """Estimate a paired mean-difference interval using deterministic resampling."""

    if len(original_scores) != len(compact_scores) or not original_scores:
        raise ValueError("paired scores must be non-empty and equal length")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not np.isfinite(maximum_allowed_drop) or maximum_allowed_drop < 0:
        raise ValueError("maximum_allowed_drop must be finite and non-negative")
    if not all(np.isfinite(value) for value in original_scores + compact_scores):
        raise ValueError("paired scores must be finite")
    differences = np.asarray(compact_scores, dtype=np.float64) - np.asarray(
        original_scores, dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    sample_count = differences.shape[0]
    bootstrap_means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        bootstrap_means[index] = differences[
            generator.integers(0, sample_count, size=sample_count)
        ].mean()
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return {
        "paired_example_count": sample_count,
        "mean_difference_compact_minus_original": float(differences.mean()),
        "confidence": confidence,
        "confidence_interval": [float(lower), float(upper)],
        "maximum_allowed_drop": maximum_allowed_drop,
        "non_inferiority_passed": float(lower) >= -maximum_allowed_drop,
        "iterations": iterations,
        "seed": seed,
    }
