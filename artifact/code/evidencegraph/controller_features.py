"""Graph-state features and marginal-information acquisition for the controller."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .controller import FEATURES
from .effective_evidence import covariance_from_dependence, effective_evidence
from .episode_inference import score_episode


def graph_state_features(
    probabilities: list[dict[str, float]],
    dependence: ArrayLike,
    *,
    remaining_budget: int,
    score_options: dict[str, Any] | None = None,
) -> np.ndarray:
    if remaining_budget < 0:
        raise ValueError("remaining budget must be non-negative")
    matrix = np.asarray(
        [
            [record["support"], record["neutral"], record["refute"]]
            for record in probabilities
        ],
        dtype=np.float64,
    )
    dep = np.asarray(dependence, dtype=np.float64)
    options = dict(score_options or {})
    score = score_episode(probabilities, dep, **options)
    feature_dependence = dep.copy()
    if options.get("use_dependence") is False:
        feature_dependence = np.eye(len(dep), dtype=np.float64)
    elif options.get("hard_dependence_threshold") is not None:
        feature_dependence = (
            feature_dependence >= float(options["hard_dependence_threshold"])
        ).astype(np.float64)
        np.fill_diagonal(feature_dependence, 1.0)
    count_covariance = covariance_from_dependence(
        feature_dependence,
        variances=np.ones(len(feature_dependence), dtype=np.float64),
        shrinkage=(
            float(options.get("shrinkage", 0.05))
            if options.get("use_dependence") is not False
            else 0.0
        ),
    )
    # The answer score uses variance-scaled precision. The controller instead
    # needs a document-count-comparable quantity, so compute precision under
    # unit marginal variance and bound numerical suppressor effects by n.
    effective_source_count = min(
        float(len(feature_dependence)),
        max(0.0, effective_evidence(count_covariance)),
    )
    off_diagonal = feature_dependence[np.triu_indices(len(feature_dependence), k=1)]
    dependence_fraction = float(np.mean(off_diagonal)) if len(off_diagonal) else 0.0
    support_mass = float(np.sum(matrix[:, 0]))
    if options.get("use_contradiction") is False:
        refute_mass = 0.0
        neutral_mass = float(np.sum(matrix[:, 1] + matrix[:, 2]))
    else:
        refute_mass = float(np.sum(matrix[:, 2]))
        neutral_mass = float(np.sum(matrix[:, 1]))
    contested_mass = support_mass + refute_mass
    conflict = (
        float(2.0 * min(support_mass, refute_mass) / contested_mass)
        if contested_mass
        else 0.0
    )
    values = np.asarray(
        [
            len(probabilities),
            effective_source_count,
            dependence_fraction,
            support_mass,
            refute_mass,
            neutral_mass,
            conflict,
            1.0 - abs(score.stance_estimate),
            remaining_budget,
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURES),) or np.any(~np.isfinite(values)):
        raise ValueError("controller feature construction produced invalid values")
    return values


def marginal_effective_candidate(
    probabilities: list[dict[str, float]],
    dependence: ArrayLike,
    selected: list[int],
    candidates: list[int],
    *,
    score_options: dict[str, Any] | None = None,
) -> tuple[int, dict[int, float]]:
    """Select the unopened source with maximum predicted information gain."""
    n_sources = len(probabilities)
    dep = np.asarray(dependence, dtype=np.float64)
    if dep.shape != (n_sources, n_sources):
        raise ValueError("dependence matrix and source probabilities must align")
    if not selected or not candidates:
        raise ValueError("selected and candidate index lists must be non-empty")
    if len(set(selected + candidates)) != len(selected) + len(candidates):
        raise ValueError("selected and candidate sources must be unique and disjoint")
    if min(selected + candidates) < 0 or max(selected + candidates) >= n_sources:
        raise ValueError("source index is out of bounds")

    def subset_score(indices: list[int]) -> float:
        subset = np.ix_(indices, indices)
        return score_episode(
            [probabilities[index] for index in indices],
            dep[subset],
            **dict(score_options or {}),
        ).effective_evidence

    current = subset_score(selected)
    gains = {
        candidate: subset_score(selected + [candidate]) - current
        for candidate in candidates
    }
    best = max(candidates, key=lambda candidate: (gains[candidate], -candidate))
    return best, gains
