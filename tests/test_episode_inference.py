from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evidencegraph.episode_inference import (
    fit_logistic_calibrator,
    load_dev_calibrator,
    score_episode,
)


SUPPORT = {"support": 0.95, "neutral": 0.03, "refute": 0.02}
REFUTE = {"support": 0.02, "neutral": 0.03, "refute": 0.95}


def test_dependence_discount_reduces_duplicate_information() -> None:
    probabilities = [SUPPORT, SUPPORT, SUPPORT]
    independent = score_episode(probabilities, np.eye(3), shrinkage=0.0)
    duplicates = score_episode(probabilities, np.ones((3, 3)), shrinkage=0.0)

    assert duplicates.stance_estimate == pytest.approx(independent.stance_estimate)
    assert duplicates.effective_evidence < independent.effective_evidence
    assert duplicates.evidence_logit < independent.evidence_logit


def test_no_contradiction_ablation_removes_negative_stance() -> None:
    dependence = np.eye(2)
    full = score_episode([REFUTE, REFUTE], dependence)
    ablated = score_episode(
        [REFUTE, REFUTE], dependence, use_contradiction=False
    )

    assert full.stance_estimate < 0.0
    assert ablated.stance_estimate == pytest.approx(0.0)


def test_hard_dependence_is_an_explicit_ablation() -> None:
    dependence = np.asarray([[1.0, 0.49], [0.49, 1.0]])
    soft = score_episode([SUPPORT, SUPPORT], dependence)
    hard = score_episode(
        [SUPPORT, SUPPORT], dependence, hard_dependence_threshold=0.5
    )

    assert hard.effective_evidence > soft.effective_evidence


def test_logistic_calibrator_learns_dev_only_probability_mapping() -> None:
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
    labels = np.asarray([0, 0, 1, 1])
    calibrator = fit_logistic_calibrator(scores, labels)
    probabilities = calibrator.predict(scores)

    assert calibrator.scale >= 0.0
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[0] < probabilities[-1]


def test_load_dev_calibrator_requires_frozen_summary_contract(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = {
        "protocol": "candidate-pool-active-seek-v1",
        "calibration_split": "dev",
        "evaluation_split": "test",
        "methods": {
            "evidencegraph_active": {
                "calibrator": {"scale": 2.0, "intercept": -0.5}
            }
        },
    }
    path.write_text(json.dumps(summary), encoding="utf-8")

    calibrator = load_dev_calibrator(path)

    assert calibrator.scale == 2.0
    summary["calibration_split"] = "test"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen dev contract"):
        load_dev_calibrator(path)
