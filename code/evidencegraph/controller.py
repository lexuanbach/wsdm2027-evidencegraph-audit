"""Small behavior-cloned controller for dependence-aware acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ACTIONS = ("answer", "find_indep", "verify", "search", "open")
FEATURES = (
    "raw_source_count",
    "effective_source_count",
    "dependence_fraction",
    "support_mass",
    "refute_mass",
    "neutral_mass",
    "conflict",
    "estimated_uncertainty",
    "remaining_budget",
)
CONTROLLER_PROTOCOL = "constructed-oracle-utility-cloning-v7-einsum-inference"


def oracle_action_utilities(features: ArrayLike) -> dict[str, float]:
    """Construction-teacher utilities from observable graph-state summaries."""

    values = np.asarray(features, dtype=float)
    if values.shape != (len(FEATURES),) or np.any(~np.isfinite(values)):
        raise ValueError(f"controller feature vector must have {len(FEATURES)} values")
    raw, effective, dependence, support, refute, neutral, conflict, uncertainty, budget = values
    if (
        raw < 1.0
        or effective < 0.0
        or not 0.0 <= dependence <= 1.0
        or min(support, refute, neutral) < 0.0
        or not 0.0 <= conflict <= 1.0
        or not 0.0 <= uncertainty <= 1.0
        or budget < 0.0
    ):
        raise ValueError("controller teacher received an invalid graph state")
    if budget <= 0.0:
        return {
            action: (0.0 if action == "answer" else float("-inf"))
            for action in ACTIONS
        }

    scale = max(1.0, raw)
    confidence = 1.0 - uncertainty
    # Requiring more than one effective source keeps a single confident copy
    # from making the answer utility look sufficient.
    sufficiency = float(np.clip(effective - 1.0, 0.0, 1.0)) * confidence
    redundancy = float(np.clip((raw - effective) / scale, 0.0, 1.0))
    dependence_dominance = 0.5 * (
        float(np.clip(dependence, 0.0, 1.0)) + redundancy
    )
    conflict_evidence = min(support, refute) / scale
    neutral_fraction = float(np.clip(neutral / scale, 0.0, 1.0))
    dominant_stance = max(support, refute) / scale
    # Every acquisition pays a fixed cost, with an additional scarcity cost
    # as the remaining three-step budget is consumed.
    acquisition_cost = 0.10 + 0.04 * (3.0 - min(budget, 3.0))

    return {
        "answer": (
            1.2 * sufficiency
            + 0.8 * confidence
            - 0.8 * conflict
            - 0.4 * neutral_fraction
        ),
        "find_indep": (
            1.3 * dependence_dominance
            + 0.8 * uncertainty
            + 0.4 * (1.0 - sufficiency)
            - acquisition_cost
        ),
        "verify": (
            1.8 * conflict
            + 0.6 * conflict_evidence
            + 0.4 * uncertainty
            - acquisition_cost
        ),
        "search": (
            1.4 * neutral_fraction
            + 0.5 * uncertainty
            + 0.2 * (1.0 - sufficiency)
            - acquisition_cost
        ),
        "open": (
            0.9 * (1.0 - sufficiency)
            + 0.4 * uncertainty
            + 0.3 * dominant_stance
            - 0.8 * neutral_fraction
            - acquisition_cost
        ),
    }


def oracle_action(features: ArrayLike) -> str:
    """Choose the deterministic maximum-utility construction-teacher action."""

    utilities = oracle_action_utilities(features)
    return max(
        ACTIONS,
        key=lambda action: (utilities[action], -ACTIONS.index(action)),
    )


@dataclass
class BehaviorClonedController:
    seed: int = 0

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=self.seed,
                            solver="liblinear",
                        )
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, features: ArrayLike, actions: ArrayLike) -> "BehaviorClonedController":
        x, y = np.asarray(features, dtype=float), np.asarray(actions, dtype=str)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or y.shape != (x.shape[0],):
            raise ValueError("invalid controller training shapes")
        unknown = set(np.unique(y)) - set(ACTIONS)
        if unknown:
            raise ValueError(f"unknown controller actions: {sorted(unknown)}")
        self.pipeline.fit(x, y)
        classifier = self.pipeline.named_steps["classifier"]
        if any(
            not np.all(np.isfinite(estimator.coef_))
            or not np.all(np.isfinite(estimator.intercept_))
            for estimator in classifier.estimators_
        ):
            raise RuntimeError("controller fit produced non-finite parameters")
        self._fitted = True
        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("controller is not fitted")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or np.any(~np.isfinite(x)):
            raise ValueError("invalid controller prediction features")
        scaled = self.pipeline.named_steps["scale"].transform(x)
        classifier = self.pipeline.named_steps["classifier"]
        coefficients = np.vstack(
            [estimator.coef_.reshape(-1) for estimator in classifier.estimators_]
        )
        intercepts = np.asarray(
            [estimator.intercept_.item() for estimator in classifier.estimators_]
        )
        # np.matmul triggers non-finite Apple Accelerate warnings on the real
        # repeated-state geometry. This is the same linear decision function
        # expressed as a stable explicit contraction.
        scores = np.einsum("ij,kj->ik", scaled, coefficients) + intercepts
        if np.any(~np.isfinite(scores)):
            raise RuntimeError("controller prediction produced non-finite scores")
        classes = np.asarray(classifier.classes_, dtype=str)
        return classes[np.argmax(scores, axis=1)]

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("cannot save an unfitted controller")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "protocol": CONTROLLER_PROTOCOL,
                "feature_names": list(FEATURES),
                "actions": list(ACTIONS),
                "seed": self.seed,
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BehaviorClonedController":
        payload = joblib.load(path)
        if (
            not isinstance(payload, dict)
            or payload.get("protocol") != CONTROLLER_PROTOCOL
            or payload.get("feature_names") != list(FEATURES)
            or payload.get("actions") != list(ACTIONS)
            or "seed" not in payload
            or "pipeline" not in payload
        ):
            raise ValueError("controller model has an incompatible feature contract")
        controller = cls(seed=int(payload["seed"]))
        controller.pipeline = payload["pipeline"]
        controller._fitted = True
        return controller
