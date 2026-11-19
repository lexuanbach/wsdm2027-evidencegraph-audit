"""Evaluation metrics with explicit denominators and paired uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike


def _binary_inputs(labels: ArrayLike, probabilities: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError("labels and probabilities must be aligned non-empty vectors")
    if not set(np.unique(y)).issubset({0, 1}) or np.any((p < 0) | (p > 1)):
        raise ValueError("expected binary labels and probabilities in [0, 1]")
    return y, p


def accuracy(labels: ArrayLike, probabilities: ArrayLike, threshold: float = 0.5) -> float:
    y, p = _binary_inputs(labels, probabilities)
    return float(np.mean((p >= threshold) == y))


def brier_score(labels: ArrayLike, probabilities: ArrayLike) -> float:
    y, p = _binary_inputs(labels, probabilities)
    return float(np.mean((p - y) ** 2))


def reliability_bins(
    labels: ArrayLike, probabilities: ArrayLike, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    y, p = _binary_inputs(labels, probabilities)
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    result: list[tuple[float, float, int]] = []
    for index in range(n_bins):
        mask = indices == index
        if np.any(mask):
            result.append((float(np.mean(p[mask])), float(np.mean(y[mask])), int(np.sum(mask))))
    return result


def expected_calibration_error(
    labels: ArrayLike, probabilities: ArrayLike, n_bins: int = 10
) -> float:
    y, p = _binary_inputs(labels, probabilities)
    bins = reliability_bins(y, p, n_bins=n_bins)
    return float(sum(count * abs(confidence - empirical) for confidence, empirical, count in bins) / y.size)


def paired_bootstrap_difference(
    labels: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float],
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    y, a = _binary_inputs(labels, first)
    _, b = _binary_inputs(labels, second)
    if n_resamples < 100:
        raise ValueError("use at least 100 bootstrap resamples")
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, y.size, y.size)
        differences[index] = metric(y[sample], a[sample]) - metric(y[sample], b[sample])
    point = metric(y, a) - metric(y, b)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(point), float(lower), float(upper)
