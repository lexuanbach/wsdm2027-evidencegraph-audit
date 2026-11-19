"""Stance-conditional effective evidence.

The shipped aggregator collapses each NLI edge to one signed scalar
``s_i = p_support - p_refute`` (``episode_inference.score_episode``) and then
runs a *single* GLS over the signed values.  Because the collapse happens
before aggregation, ``w_eff = 1' Sigma^+ 1`` is stance-agnostic: it measures
how many independent *sources* the pool contains, not how much independent
support and how much independent refutation it contains.

This module tests whether computing a separate ``w_eff`` on each side and
combining the two through a calibrated comparative model repairs the
``support_refute_mix`` failure.  It implements exactly that, keeping
every other frozen ingredient identical (edge variances, shrinkage, PSD
projection, pseudoinverse ``rcond``, dev-only calibration).

Definitions.  With ``s_i = p_support,i - p_refute,i`` and an informativeness
threshold ``tau >= 0``:

* support side  ``S+ = {i : s_i >= tau}``,
* refute side   ``S- = {i : s_i <= -tau}``,
* everything else is dropped from both sides.

For each side the dependence matrix is restricted to that index set, the
covariance is rebuilt with the *same* ``covariance_from_dependence`` call the
production aggregator uses, and ``aggregate_stances`` is run on the
*magnitudes* ``|s_i|`` (both sides therefore return a non-negative mean).  The
two side statistics are

    ``E+ = theta+ * sqrt(w+)``  and  ``E- = theta- * sqrt(w-)``,

the direct per-stance analogue of the production ``evidence_logit``.  An empty
side contributes ``theta = w = E = 0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .effective_evidence import aggregate_stances, covariance_from_dependence
from .episode_inference import edge_variances

NLI_LABELS = ("support", "neutral", "refute")


@dataclass(frozen=True)
class StanceConditionalScore:
    """Per-stance aggregates for one episode."""

    support_stance: float
    support_effective_evidence: float
    support_evidence: float
    support_count: int
    refute_stance: float
    refute_effective_evidence: float
    refute_evidence: float
    refute_count: int
    dropped_count: int
    raw_source_count: int

    @property
    def difference(self) -> float:
        return self.support_evidence - self.refute_evidence

    def to_dict(self) -> dict[str, float | int]:
        record = asdict(self)
        record["evidence_difference"] = self.difference
        return record


def _probability_matrix(probabilities: list[dict[str, float]]) -> np.ndarray:
    matrix = np.asarray(
        [[float(record[label]) for label in NLI_LABELS] for record in probabilities],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] == 0:
        raise ValueError("at least one NLI probability record is required")
    if np.any(~np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("NLI probabilities must be finite values in [0, 1]")
    return matrix


def _side_statistics(
    magnitudes: np.ndarray,
    dependence: np.ndarray,
    indices: np.ndarray,
    *,
    shrinkage: float,
) -> tuple[float, float]:
    """GLS mean and effective evidence of one stance side."""

    if indices.size == 0:
        return 0.0, 0.0
    values = magnitudes[indices]
    sub = dependence[np.ix_(indices, indices)]
    covariance = covariance_from_dependence(
        sub, variances=edge_variances(values), shrinkage=shrinkage
    )
    estimate, effective = aggregate_stances(values, covariance)
    return float(estimate), float(effective)


def score_episode_by_stance(
    probabilities: list[dict[str, float]],
    dependence: ArrayLike,
    *,
    use_dependence: bool = True,
    hard_dependence_threshold: float | None = None,
    shrinkage: float = 0.05,
    threshold: float = 0.0,
) -> StanceConditionalScore:
    """Aggregate support and refute edges separately."""

    matrix = _probability_matrix(probabilities)
    n_sources = matrix.shape[0]
    dependence_array = np.asarray(dependence, dtype=np.float64)
    if dependence_array.shape != (n_sources, n_sources):
        raise ValueError("dependence matrix and NLI records must align")
    if not np.allclose(dependence_array, dependence_array.T, atol=1e-8):
        raise ValueError("dependence matrix must be symmetric")
    dependence_array = np.clip(dependence_array.copy(), 0.0, 1.0)
    np.fill_diagonal(dependence_array, 1.0)
    if not use_dependence:
        dependence_array = np.eye(n_sources, dtype=np.float64)
    elif hard_dependence_threshold is not None:
        if not 0.0 < hard_dependence_threshold < 1.0:
            raise ValueError("hard dependence threshold must be in (0, 1)")
        dependence_array = (dependence_array >= hard_dependence_threshold).astype(np.float64)
        np.fill_diagonal(dependence_array, 1.0)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    stances = matrix[:, 0] - matrix[:, 2]
    support_index = np.flatnonzero(stances >= max(threshold, np.nextafter(0.0, 1.0)))
    refute_index = np.flatnonzero(-stances >= max(threshold, np.nextafter(0.0, 1.0)))
    magnitudes = np.abs(stances)
    effective_shrinkage = shrinkage if use_dependence else 0.0
    support_theta, support_w = _side_statistics(
        magnitudes, dependence_array, support_index, shrinkage=effective_shrinkage
    )
    refute_theta, refute_w = _side_statistics(
        magnitudes, dependence_array, refute_index, shrinkage=effective_shrinkage
    )
    return StanceConditionalScore(
        support_stance=support_theta,
        support_effective_evidence=support_w,
        support_evidence=float(support_theta * np.sqrt(support_w)),
        support_count=int(support_index.size),
        refute_stance=refute_theta,
        refute_effective_evidence=refute_w,
        refute_evidence=float(refute_theta * np.sqrt(refute_w)),
        refute_count=int(refute_index.size),
        dropped_count=int(n_sources - support_index.size - refute_index.size),
        raw_source_count=n_sources,
    )


# --------------------------------------------------------------------------
# Calibrated comparative models.  Every parameter below is fitted on dev only.
# --------------------------------------------------------------------------


COMPARATIVE_FORMS = ("difference_2p", "asymmetric_3p", "generous_5p")

_FEATURES: dict[str, tuple[str, ...]] = {
    "difference_2p": ("evidence_difference",),
    "asymmetric_3p": ("support_evidence", "refute_evidence"),
    "generous_5p": (
        "support_evidence",
        "refute_evidence",
        "sqrt_support_effective_evidence",
        "sqrt_refute_effective_evidence",
    ),
}

# Sign constraints keep the comparative model monotone in the intended
# direction: more independent support raises the probability, more
# independent refutation lowers it.  ``generous_5p`` is deliberately
# unconstrained so the stance-split proposal gets its best possible showing.
_BOUNDS: dict[str, list[tuple[float, float]]] = {
    "difference_2p": [(0.0, 100.0)],
    "asymmetric_3p": [(0.0, 100.0), (-100.0, 0.0)],
    "generous_5p": [(-100.0, 100.0)] * 4,
}


def comparative_features(record: dict[str, Any], form: str) -> np.ndarray:
    if form not in COMPARATIVE_FORMS:
        raise ValueError(f"unknown comparative form: {form}")
    values = []
    for name in _FEATURES[form]:
        if name.startswith("sqrt_"):
            values.append(float(np.sqrt(max(0.0, float(record[name[len("sqrt_"):]])))))
        else:
            values.append(float(record[name]))
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class ComparativeModel:
    """Dev-fitted logistic over per-stance evidence features."""

    form: str
    coefficients: tuple[float, ...]
    intercept: float

    def logit(self, features: np.ndarray) -> np.ndarray:
        design = np.atleast_2d(np.asarray(features, dtype=np.float64))
        beta = np.asarray(self.coefficients, dtype=np.float64)
        return np.clip(design @ beta + self.intercept, -40.0, 40.0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(features)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "features": list(_FEATURES[self.form]),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }


def fit_comparative_model(
    features: ArrayLike, labels: ArrayLike, *, form: str
) -> ComparativeModel:
    """Fit the comparative model on one split (dev) by log-loss."""

    if form not in COMPARATIVE_FORMS:
        raise ValueError(f"unknown comparative form: {form}")
    design = np.atleast_2d(np.asarray(features, dtype=np.float64))
    targets = np.asarray(labels, dtype=np.int64)
    k = design.shape[1]
    if k != len(_FEATURES[form]) or targets.shape != (design.shape[0],):
        raise ValueError("comparative design matrix and labels must align")
    if set(np.unique(targets)) != {0, 1} or np.any(~np.isfinite(design)):
        raise ValueError("comparative fitting requires finite features and both labels")

    def objective(parameters: np.ndarray) -> float:
        logits = np.clip(design @ parameters[:k] + parameters[k], -40.0, 40.0)
        return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))

    bounds = list(_BOUNDS[form]) + [(-20.0, 20.0)]
    x0 = np.zeros(k + 1, dtype=np.float64)
    for index, (low, high) in enumerate(_BOUNDS[form]):
        x0[index] = 1.0 if high > 0.0 else -1.0
    result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"comparative model optimization failed: {result.message}")
    return ComparativeModel(
        form=form,
        coefficients=tuple(float(value) for value in result.x[:k]),
        intercept=float(result.x[k]),
    )


def log_loss(labels: ArrayLike, probabilities: ArrayLike) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def both_stances_present(
    probabilities: list[dict[str, float]], *, threshold: float = 0.0
) -> bool:
    """True when the pool contains at least one support and one refute edge."""

    matrix = _probability_matrix(probabilities)
    stances = matrix[:, 0] - matrix[:, 2]
    cut = max(threshold, np.nextafter(0.0, 1.0))
    return bool(np.any(stances >= cut) and np.any(-stances >= cut))
