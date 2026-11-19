from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.controller import FEATURES
from evidencegraph.controller_features import (
    graph_state_features,
    marginal_effective_candidate,
)


SUPPORT = {"support": 0.95, "neutral": 0.03, "refute": 0.02}
REFUTE = {"support": 0.02, "neutral": 0.03, "refute": 0.95}


def test_graph_state_features_follow_controller_contract() -> None:
    features = graph_state_features(
        [SUPPORT, REFUTE], np.asarray([[1.0, 0.2], [0.2, 1.0]]), remaining_budget=2
    )

    assert features.shape == (len(FEATURES),)
    assert features[0] == 2
    assert features[6] > 0.9
    assert features[-1] == 2


def test_controller_effective_count_is_comparable_to_raw_source_count() -> None:
    copies = graph_state_features(
        [SUPPORT, SUPPORT, SUPPORT],
        np.ones((3, 3)),
        remaining_budget=1,
    )
    independent = graph_state_features(
        [SUPPORT, SUPPORT, SUPPORT],
        np.eye(3),
        remaining_budget=1,
    )

    assert FEATURES[1] == "effective_source_count"
    assert 1.0 <= copies[1] < 1.1
    assert independent[1] == pytest.approx(3.0)
    assert copies[0] - copies[1] > 1.5


def test_marginal_acquisition_prefers_independent_source_over_copy() -> None:
    probabilities = [SUPPORT, SUPPORT, SUPPORT]
    dependence = np.asarray(
        [
            [1.0, 0.99, 0.0],
            [0.99, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    best, gains = marginal_effective_candidate(
        probabilities, dependence, selected=[0], candidates=[1, 2]
    )

    assert best == 2
    assert gains[2] > gains[1]


def test_marginal_acquisition_rejects_overlapping_state_sets() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        marginal_effective_candidate(
            [SUPPORT, SUPPORT], np.eye(2), selected=[0], candidates=[0]
        )
