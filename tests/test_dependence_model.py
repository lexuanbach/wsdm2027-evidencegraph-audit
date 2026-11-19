from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest

from evidencegraph.dependence_features import FEATURE_NAMES
from evidencegraph.dependence_model import DependenceEstimator


def test_dependence_fit_and_prediction_avoid_accelerate_matmul_warnings(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(103)
    features = rng.normal(size=(2504, len(FEATURE_NAMES)))
    labels = np.asarray([index % 2 for index in range(len(features))])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        estimator = DependenceEstimator(seed=103).fit(features, labels)
        probabilities = estimator.predict_proba(features[:100])
        model_path = tmp_path / "dependence.joblib"
        estimator.save(model_path)
        reloaded = DependenceEstimator.load(model_path)
        reloaded_probabilities = reloaded.predict_proba(features[:100])

    classifier = estimator.pipeline.named_steps["classifier"]
    assert classifier.estimator.solver == "liblinear"
    assert probabilities.shape == (100,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(reloaded_probabilities, probabilities)


def test_dependence_model_rejects_stale_feature_contract(tmp_path: Path) -> None:
    rng = np.random.default_rng(103)
    features = rng.normal(size=(100, len(FEATURE_NAMES)))
    labels = np.asarray([index % 2 for index in range(len(features))])
    model_path = tmp_path / "dependence.joblib"
    DependenceEstimator(seed=103).fit(features, labels).save(model_path)
    payload = joblib.load(model_path)
    payload["feature_names"] = payload["feature_names"][:-1]
    joblib.dump(payload, model_path)

    with pytest.raises(ValueError, match="incompatible feature contract"):
        DependenceEstimator.load(model_path)


def test_dependence_model_rejects_wrong_feature_dimension() -> None:
    features = np.zeros((10, len(FEATURE_NAMES) - 1))
    labels = np.asarray([index % 2 for index in range(len(features))])

    with pytest.raises(ValueError, match="features must be 2-D"):
        DependenceEstimator(seed=103).fit(features, labels)
