from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest

from evidencegraph.controller import (
    ACTIONS,
    BehaviorClonedController,
    CONTROLLER_PROTOCOL,
    FEATURES,
    oracle_action,
    oracle_action_utilities,
)


def test_oracle_seeks_independent_evidence_when_duplicates_dominate() -> None:
    features = [10, 1.2, 0.88, 8, 0.2, 0.1, 0.0, 0.4, 3]
    assert oracle_action(features) == "find_indep"


def test_oracle_answers_when_confident_or_out_of_budget() -> None:
    assert oracle_action([3, 3, 0, 2.8, 0.1, 0.1, 0, 0.05, 2]) == "answer"
    assert oracle_action([1, 1, 0, 0.4, 0.2, 0.4, 0.2, 0.5, 0]) == "answer"


def test_oracle_verifies_conflict_before_seeking_around_dependence() -> None:
    features = [8, 2, 0.8, 3.0, 2.5, 0.5, 0.9, 0.8, 3]
    assert oracle_action(features) == "verify"


def test_oracle_utility_exposes_every_frozen_decision_term() -> None:
    features = [10, 1.2, 0.88, 8, 0.2, 0.1, 0.0, 0.4, 3]
    utilities = oracle_action_utilities(features)

    assert set(utilities) == set(ACTIONS)
    assert utilities["find_indep"] > utilities["answer"]
    assert oracle_action_utilities([1, 1, 0, 0.4, 0.2, 0.4, 0.2, 0.5, 0]) == {
        "answer": 0.0,
        "find_indep": float("-inf"),
        "verify": float("-inf"),
        "search": float("-inf"),
        "open": float("-inf"),
    }


def test_controller_fit_avoids_accelerate_multinomial_matmul_warnings() -> None:
    rng = np.random.default_rng(103)
    features = rng.normal(size=(2504, 9))
    actions = np.asarray([ACTIONS[index % len(ACTIONS)] for index in range(len(features))])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        controller = BehaviorClonedController(seed=103).fit(features, actions)
        predictions = controller.predict(features)

    classifier = controller.pipeline.named_steps["classifier"]
    assert classifier.estimator.solver == "liblinear"
    assert all(np.isfinite(estimator.coef_).all() for estimator in classifier.estimators_)
    assert predictions.shape == actions.shape


def test_controller_model_payload_binds_protocol_features_and_actions(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(103)
    features = rng.normal(size=(100, len(FEATURES)))
    actions = np.asarray([ACTIONS[index % len(ACTIONS)] for index in range(100)])
    model_path = tmp_path / "controller.joblib"
    BehaviorClonedController(seed=103).fit(features, actions).save(model_path)

    payload = joblib.load(model_path)
    assert payload["protocol"] == CONTROLLER_PROTOCOL
    assert payload["feature_names"] == list(FEATURES)
    assert payload["actions"] == list(ACTIONS)
    assert BehaviorClonedController.load(model_path).predict(features).shape == (100,)

    for key, wrong in (
        ("protocol", "stale"),
        ("feature_names", list(FEATURES[:-1])),
        ("actions", list(reversed(ACTIONS))),
    ):
        changed = dict(payload)
        changed[key] = wrong
        stale_path = tmp_path / f"stale-{key}.joblib"
        joblib.dump(changed, stale_path)
        with pytest.raises(ValueError, match="incompatible feature contract"):
            BehaviorClonedController.load(stale_path)
