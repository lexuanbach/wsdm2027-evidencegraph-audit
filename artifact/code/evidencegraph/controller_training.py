"""Leakage-safe controller demonstrations from constructed graph states."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .controller import (
    ACTIONS,
    CONTROLLER_PROTOCOL,
    FEATURES,
    BehaviorClonedController,
    oracle_action,
)
from .controller_features import graph_state_features


def constructed_nli_probabilities(source: dict[str, Any]) -> dict[str, float]:
    """Convert construction stance into a soft oracle edge for demonstrations."""
    stance = float(source["stance"])
    confidence = float(source["stance_confidence"])
    if not -1.0 <= stance <= 1.0 or not 0.0 <= confidence <= 1.0:
        raise ValueError("constructed source has invalid stance metadata")
    mass = confidence * abs(stance)
    if stance > 0.0:
        return {"support": mass, "neutral": 1.0 - mass, "refute": 0.0}
    if stance < 0.0:
        return {"support": 0.0, "neutral": 1.0 - mass, "refute": mass}
    return {"support": 0.0, "neutral": 1.0, "refute": 0.0}


def constructed_dependence(sources: list[dict[str, Any]]) -> np.ndarray:
    origins = [str(source["origin_id"]) for source in sources]
    return np.asarray(
        [[float(left == right) for right in origins] for left in origins],
        dtype=np.float64,
    )


def controller_demonstrations(
    episodes: list[dict[str, Any]],
    split: str,
    *,
    maximum_budget: int = 3,
    include_metadata: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if maximum_budget < 1:
        raise ValueError("maximum controller budget must be positive")
    selected = [episode for episode in episodes if episode["split"] == split]
    state_count = sum(
        len(episode["sources"]) * (maximum_budget + 1) for episode in selected
    )
    if state_count < 1:
        raise ValueError(f"no controller demonstrations for split {split}")
    features = np.empty((state_count, len(FEATURES)), dtype=np.float64)
    actions = np.empty(state_count, dtype=f"<U{max(map(len, ACTIONS))}")
    metadata: list[dict[str, Any]] = []
    row = 0
    for episode in selected:
        sources = list(episode["sources"])
        probabilities = [constructed_nli_probabilities(source) for source in sources]
        dependence = constructed_dependence(sources)
        for prefix in range(1, len(sources) + 1):
            index = list(range(prefix))
            subset = np.ix_(index, index)
            base_state = graph_state_features(
                probabilities[:prefix],
                dependence[subset],
                remaining_budget=0,
            )
            for remaining_budget in range(maximum_budget + 1):
                state = base_state.copy()
                state[-1] = float(remaining_budget)
                action = oracle_action(state)
                features[row] = state
                actions[row] = action
                if include_metadata:
                    metadata.append(
                        {
                            "claim_id": str(episode["claim_id"]),
                            "prefix": prefix,
                            "remaining_budget": remaining_budget,
                            "action": action,
                        }
                    )
                row += 1
    if row != state_count or (include_metadata and len(metadata) != state_count):
        raise RuntimeError("controller-state preallocation count drifted")
    return features, actions, metadata


def train_and_evaluate_controller(
    episodes: list[dict[str, Any]],
    *,
    seed: int = 103,
    maximum_budget: int = 3,
    allow_missing_actions: bool = False,
) -> tuple[BehaviorClonedController, dict[str, Any]]:
    train_x, train_y, _ = controller_demonstrations(
        episodes,
        "train",
        maximum_budget=maximum_budget,
        include_metadata=False,
    )
    dev_x, dev_y, _ = controller_demonstrations(
        episodes,
        "dev",
        maximum_budget=maximum_budget,
        include_metadata=False,
    )
    observed = set(np.unique(train_y))
    missing_actions = sorted(set(ACTIONS) - observed)
    if missing_actions and not allow_missing_actions:
        raise ValueError(f"training demonstrations lack actions: {missing_actions}")
    controller = BehaviorClonedController(seed=seed).fit(train_x, train_y)
    del train_x
    predictions = controller.predict(dev_x)
    summary = {
        "protocol": CONTROLLER_PROTOCOL,
        "seed": seed,
        "maximum_budget": maximum_budget,
        "training_split": "train",
        "evaluation_split": "dev",
        "deployed_feature_names": list(FEATURES),
        "construction_target_fields": ["stance", "stance_confidence", "origin_id"],
        "teacher_utility_terms": [
            "answer_sufficiency",
            "uncertainty",
            "independent_evidence_gain",
            "conflict_resolution",
            "acquisition_cost",
        ],
        "protected_fields_excluded_from_deployed_features": [
            "label",
            "gold_label",
            "family",
            "origin_id",
            "corruption",
            "stance",
        ],
        "missing_teacher_actions": missing_actions,
        "allow_missing_actions": bool(allow_missing_actions),
        "train_states": int(len(train_y)),
        "dev_states": int(len(dev_y)),
        "train_action_counts": dict(Counter(train_y)),
        "dev_action_counts": dict(Counter(dev_y)),
        "dev_accuracy": float(accuracy_score(dev_y, predictions)),
        "dev_macro_f1": float(
            f1_score(dev_y, predictions, labels=list(ACTIONS), average="macro", zero_division=0)
        ),
        "dev_prediction_counts": dict(Counter(predictions)),
    }
    return controller, summary
