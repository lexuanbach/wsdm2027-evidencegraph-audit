"""Small calibrated classifier for probabilistic dependence edges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dependence_features import DEPENDENCE_SCORER_PROTOCOL, FEATURE_NAMES


class StableLogisticRegression(LogisticRegression):
    """Logistic regression with an Apple-Accelerate-safe decision function."""

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "coef_"):
            raise RuntimeError("stable logistic estimator is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coef_.shape[1] or np.any(~np.isfinite(x)):
            raise ValueError("invalid stable logistic prediction features")
        scores = np.einsum("ij,kj->ik", x, self.coef_) + self.intercept_
        if np.any(~np.isfinite(scores)):
            raise RuntimeError("stable logistic prediction produced non-finite scores")
        return scores[:, 0] if scores.shape[1] == 1 else scores


@dataclass
class DependenceEstimator:
    seed: int = 0

    def __post_init__(self) -> None:
        base = StableLogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=self.seed,
            solver="liblinear",
        )
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
            ]
        )
        self._fitted = False

    def fit(self, features: ArrayLike, labels: ArrayLike) -> "DependenceEstimator":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if (
            x.ndim != 2
            or x.shape[1] != len(FEATURE_NAMES)
            or y.shape != (x.shape[0],)
        ):
            raise ValueError("features must be 2-D and labels must align")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("training data must contain both classes")
        self.pipeline.fit(x, y)
        classifier = self.pipeline.named_steps["classifier"]
        for calibrated in classifier.calibrated_classifiers_:
            estimator = calibrated.estimator
            if not np.all(np.isfinite(estimator.coef_)) or not np.all(
                np.isfinite(estimator.intercept_)
            ):
                raise RuntimeError("dependence fit produced non-finite parameters")
            if any(
                not np.isfinite(calibrator.a_) or not np.isfinite(calibrator.b_)
                for calibrator in calibrated.calibrators
            ):
                raise RuntimeError("dependence calibration produced non-finite parameters")
        self._fitted = True
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("dependence estimator is not fitted")
        x = np.asarray(features, dtype=np.float64)
        if (
            x.ndim != 2
            or x.shape[1] != len(FEATURE_NAMES)
            or np.any(~np.isfinite(x))
        ):
            raise ValueError("invalid dependence prediction features")
        probabilities = self.pipeline.predict_proba(x)[:, 1]
        if probabilities.shape != (len(x),) or np.any(~np.isfinite(probabilities)):
            raise RuntimeError("dependence prediction produced invalid probabilities")
        return probabilities

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("cannot save an unfitted estimator")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "protocol": DEPENDENCE_SCORER_PROTOCOL,
                "feature_names": list(FEATURE_NAMES),
                "seed": self.seed,
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "DependenceEstimator":
        payload = joblib.load(path)
        if (
            payload.get("protocol") != DEPENDENCE_SCORER_PROTOCOL
            or payload.get("feature_names") != list(FEATURE_NAMES)
        ):
            raise ValueError("dependence model has an incompatible feature contract")
        estimator = cls(seed=int(payload["seed"]))
        estimator.pipeline = payload["pipeline"]
        estimator._fitted = True
        return estimator
