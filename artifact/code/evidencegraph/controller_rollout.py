"""Deterministic candidate-pool rollouts for the learned controller."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from .controller import ACTIONS
from .controller_features import graph_state_features, marginal_effective_candidate
from .episode_inference import score_episode


class Controller(Protocol):
    def predict(self, features: ArrayLike) -> np.ndarray: ...


def rollout_controller(
    probabilities: list[dict[str, float]],
    dependence: ArrayLike,
    controller: Controller,
    *,
    initial_sources: int = 5,
    action_budget: int = 3,
    enable_independent_seek: bool = True,
    score_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_sources = len(probabilities)
    dep = np.asarray(dependence, dtype=np.float64)
    if dep.shape != (n_sources, n_sources) or n_sources < initial_sources:
        raise ValueError("controller rollout inputs or initial source count are invalid")
    if initial_sources < 1 or action_budget < 0:
        raise ValueError("controller source and action budgets must be non-negative")
    selected = list(range(initial_sources))
    candidates = list(range(initial_sources, n_sources))
    actions = []
    acquisitions = []
    remaining = action_budget
    while remaining > 0:
        subset = np.ix_(selected, selected)
        features = graph_state_features(
            [probabilities[index] for index in selected],
            dep[subset],
            remaining_budget=remaining,
            score_options=score_options,
        )
        action = str(controller.predict(features.reshape(1, -1))[0])
        if action not in ACTIONS:
            raise ValueError(f"controller emitted unknown action: {action}")
        actions.append(action)
        if action == "answer":
            break
        remaining -= 1
        if action == "verify":
            # Edge scorers are frozen in this controlled candidate-pool run;
            # verification consumes a step but cannot silently replace a score.
            continue
        if not candidates:
            break
        if action == "find_indep" and enable_independent_seek:
            candidate, _ = marginal_effective_candidate(
                probabilities,
                dep,
                selected,
                candidates,
                score_options=score_options,
            )
        else:
            candidate = candidates[0]
        selected.append(candidate)
        candidates.remove(candidate)
        acquisitions.append(
            {
                "action": action,
                "source_index": candidate,
                "rank_skipped": int(candidate - min([candidate, *candidates], default=candidate)),
            }
        )
    subset = np.ix_(selected, selected)
    final = score_episode(
        [probabilities[index] for index in selected],
        dep[subset],
        **dict(score_options or {}),
    )
    return {
        "selected_source_indices": selected,
        "actions": actions,
        "acquisitions": acquisitions,
        "find_indep_count": actions.count("find_indep"),
        "search_count": actions.count("search") + actions.count("find_indep"),
        "documents_opened": len(selected),
        "remaining_budget": remaining,
        **final.to_dict(),
    }
