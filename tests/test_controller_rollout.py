from __future__ import annotations

import numpy as np

from evidencegraph.controller_rollout import rollout_controller


SUPPORT = {"support": 0.95, "neutral": 0.03, "refute": 0.02}


class FixedController:
    def __init__(self, actions):
        self.actions = iter(actions)

    def predict(self, features):
        return np.asarray([next(self.actions)])


def test_find_indep_rollout_skips_dependent_next_rank() -> None:
    probabilities = [SUPPORT] * 4
    dependence = np.asarray(
        [
            [1.0, 0.99, 0.99, 0.0],
            [0.99, 1.0, 0.99, 0.0],
            [0.99, 0.99, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rollout = rollout_controller(
        probabilities,
        dependence,
        FixedController(["find_indep"]),
        initial_sources=2,
        action_budget=1,
    )

    assert rollout["selected_source_indices"] == [0, 1, 3]
    assert rollout["find_indep_count"] == 1


def test_no_active_seek_ablation_opens_next_rank() -> None:
    probabilities = [SUPPORT] * 4
    dependence = np.eye(4)
    rollout = rollout_controller(
        probabilities,
        dependence,
        FixedController(["find_indep"]),
        initial_sources=2,
        action_budget=1,
        enable_independent_seek=False,
    )

    assert rollout["selected_source_indices"] == [0, 1, 2]


def test_verify_consumes_budget_without_mutating_frozen_edges() -> None:
    rollout = rollout_controller(
        [SUPPORT, SUPPORT],
        np.eye(2),
        FixedController(["verify"]),
        initial_sources=2,
        action_budget=1,
    )

    assert rollout["documents_opened"] == 2
    assert rollout["remaining_budget"] == 0
