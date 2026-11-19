"""Dependence-aware episode inference and dev-set calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .effective_evidence import aggregate_stances, covariance_from_dependence


NLI_LABELS = ("support", "neutral", "refute")


@dataclass(frozen=True)
class EpisodeScore:
    stance_estimate: float
    effective_evidence: float
    evidence_logit: float
    raw_source_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class LogisticCalibrator:
    scale: float
    intercept: float

    def predict(self, evidence_logits: ArrayLike) -> np.ndarray:
        values = np.asarray(evidence_logits, dtype=np.float64)
        logits = np.clip(self.scale * values + self.intercept, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_dev_calibrator(
    path: Path,
    *,
    section: str = "methods",
    key: str = "evidencegraph_active",
) -> LogisticCalibrator:
    """Load a matched frozen EvidenceGraph-20K dev calibrator."""

    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    try:
        record = summary[section][key]["calibrator"]
        scale = float(record["scale"])
        intercept = float(record["intercept"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"calibration summary lacks the dev-fitted {section}/{key} calibrator"
        ) from error
    expected_protocol = {
        "methods": "candidate-pool-active-seek-v1",
        "variants": "evidencegraph-episode-inference-v1",
    }.get(section)
    if (
        expected_protocol is None
        or summary.get("protocol") != expected_protocol
        or summary.get("calibration_split") != "dev"
        or summary.get("evaluation_split") != "test"
        or not math.isfinite(scale)
        or not math.isfinite(intercept)
        or scale < 0.0
    ):
        raise ValueError("calibration summary violates the frozen dev contract")
    return LogisticCalibrator(scale=scale, intercept=intercept)


def _probability_matrix(probabilities: list[dict[str, float]]) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one NLI probability record is required")
    if any(set(record) != set(NLI_LABELS) for record in probabilities):
        raise ValueError(f"NLI probabilities must contain exactly {NLI_LABELS}")
    matrix = np.asarray(
        [[float(record[label]) for label in NLI_LABELS] for record in probabilities],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("NLI probabilities must be finite values in [0, 1]")
    if not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-5):
        raise ValueError("NLI probabilities must sum to one")
    return matrix


EDGE_VARIANCE_FLOOR = 0.05


def edge_variances(stances: ArrayLike) -> np.ndarray:
    """Observation variance of one NLI edge from its signed stance.

    Uncertain/neutral edges receive larger observation variance.  The floor
    prevents a single saturated checkpoint probability from gaining infinite
    precision.  This is the single definition of the rule; audits must import
    it rather than restate the formula.
    """

    values = np.asarray(stances, dtype=np.float64)
    return np.maximum(EDGE_VARIANCE_FLOOR, 1.0 - np.abs(values))


def score_episode(
    probabilities: list[dict[str, float]],
    dependence: ArrayLike,
    *,
    use_dependence: bool = True,
    use_contradiction: bool = True,
    use_effective_evidence: bool = True,
    hard_dependence_threshold: float | None = None,
    shrinkage: float = 0.05,
    scorer_variance: float = 0.0,
    ridge: float = 0.0,
    ridge_mode: str = "covariance",
) -> EpisodeScore:
    """Aggregate source edges into a signed score without fitting on test data."""
    matrix = _probability_matrix(probabilities)
    n_sources = len(matrix)
    dependence_array = np.asarray(dependence, dtype=np.float64)
    if dependence_array.shape != (n_sources, n_sources):
        raise ValueError("dependence matrix and NLI records must align")
    if np.any(~np.isfinite(dependence_array)):
        raise ValueError("dependence matrix contains non-finite values")
    if not np.allclose(dependence_array, dependence_array.T, atol=1e-8):
        raise ValueError("dependence matrix must be symmetric")
    if np.any((dependence_array < 0.0) | (dependence_array > 1.0)):
        raise ValueError("dependence probabilities must be in [0, 1]")
    dependence_array = dependence_array.copy()
    np.fill_diagonal(dependence_array, 1.0)
    if not use_dependence:
        dependence_array = np.eye(n_sources, dtype=np.float64)
    elif hard_dependence_threshold is not None:
        if not 0.0 < hard_dependence_threshold < 1.0:
            raise ValueError("hard dependence threshold must be in (0, 1)")
        dependence_array = (
            dependence_array >= hard_dependence_threshold
        ).astype(np.float64)
        np.fill_diagonal(dependence_array, 1.0)
    stances = matrix[:, 0] - matrix[:, 2]
    if not use_contradiction:
        stances = np.maximum(stances, 0.0)
    variances = edge_variances(stances)
    covariance = covariance_from_dependence(
        dependence_array,
        variances=variances,
        shrinkage=shrinkage if use_dependence else 0.0,
        scorer_variance=scorer_variance,
    )
    estimate, effective = aggregate_stances(
        stances, covariance, ridge=ridge, ridge_mode=ridge_mode
    )
    information_scale = np.sqrt(effective) if use_effective_evidence else 1.0
    return EpisodeScore(
        stance_estimate=float(estimate),
        effective_evidence=float(effective),
        evidence_logit=float(estimate * information_scale),
        raw_source_count=n_sources,
    )


def fit_logistic_calibrator(
    evidence_logits: ArrayLike, labels: ArrayLike
) -> LogisticCalibrator:
    """Fit two dev-only calibration parameters with a non-negative slope."""
    scores = np.asarray(evidence_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or targets.shape != scores.shape or scores.size < 2:
        raise ValueError("calibration scores and labels must be aligned vectors")
    if set(np.unique(targets)) != {0, 1} or np.any(~np.isfinite(scores)):
        raise ValueError("calibration requires finite scores and both binary labels")

    def objective(parameters: np.ndarray) -> float:
        scale, intercept = parameters
        logits = np.clip(scale * scores + intercept, -40.0, 40.0)
        return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))

    result = minimize(
        objective,
        x0=np.asarray([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=((0.0, 100.0), (-20.0, 20.0)),
    )
    if not result.success:
        raise RuntimeError(f"calibrator optimization failed: {result.message}")
    return LogisticCalibrator(scale=float(result.x[0]), intercept=float(result.x[1]))


def calibrated_record(score: EpisodeScore, calibrator: LogisticCalibrator) -> dict[str, Any]:
    probability = float(calibrator.predict([score.evidence_logit])[0])
    return {**score.to_dict(), "probability_true": probability}
