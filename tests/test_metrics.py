from __future__ import annotations

import pytest

from evidencegraph.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


def test_perfect_predictions_have_perfect_metrics() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.0, 0.0, 1.0, 1.0]
    assert accuracy(labels, probabilities) == 1.0
    assert brier_score(labels, probabilities) == 0.0
    assert expected_calibration_error(labels, probabilities) == 0.0


def test_reliability_preserves_denominator() -> None:
    bins = reliability_bins([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=5)
    assert sum(count for _, _, count in bins) == 4
    # Equal-width bins are half-open except the final bin, so 0.2 begins bin 2.
    assert bins[0] == pytest.approx((0.1, 0.0, 1))
