"""Memory-bounded helpers for deterministic claim-cluster bootstraps."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


DEFAULT_WEIGHT_BATCH_SIZE = 128


def multinomial_weight_batches(
    rng: np.random.Generator,
    n_items: int,
    n_resamples: int,
    *,
    batch_size: int = DEFAULT_WEIGHT_BATCH_SIZE,
) -> Iterator[np.ndarray]:
    """Yield the exact seeded multinomial bootstrap weights in bounded batches."""

    if n_items < 1 or n_resamples < 1 or batch_size < 1:
        raise ValueError("bootstrap item, resample, and batch counts must be positive")
    probabilities = np.full(n_items, 1.0 / n_items, dtype=np.float64)
    for offset in range(0, n_resamples, batch_size):
        yield rng.multinomial(
            n_items,
            probabilities,
            size=min(batch_size, n_resamples - offset),
        )
