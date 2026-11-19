from __future__ import annotations

import numpy as np
import pytest

from evidencegraph.controller_training import (
    constructed_dependence,
    constructed_nli_probabilities,
    controller_demonstrations,
    train_and_evaluate_controller,
)
from evidencegraph.controller_features import graph_state_features
from evidencegraph.controller import CONTROLLER_PROTOCOL


def _source(source_id, origin, stance):
    return {
        "source_id": source_id,
        "origin_id": origin,
        "stance": stance,
        "stance_confidence": 0.95 if stance else 0.5,
    }


def test_constructed_edges_preserve_stance_and_origin() -> None:
    support = _source("a", "o", 1.0)
    neutral = _source("b", "n", 0.0)
    probabilities = constructed_nli_probabilities(support)
    assert probabilities["support"] == pytest.approx(0.95)
    assert probabilities["neutral"] == pytest.approx(0.05)
    assert probabilities["refute"] == 0.0
    assert constructed_nli_probabilities(neutral)["neutral"] == 1.0
    dependence = constructed_dependence([support, _source("c", "o", -1.0), neutral])
    assert np.array_equal(
        dependence, np.asarray([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    )


def test_controller_demonstrations_are_split_scoped_and_budgeted() -> None:
    episodes = [
        {
            "claim_id": "train-case",
            "split": "train",
            "sources": [_source("a", "o", 1.0), _source("b", "o", 1.0)],
        },
        {
            "claim_id": "dev-case",
            "split": "dev",
            "sources": [_source("c", "p", -1.0)],
        },
    ]
    features, actions, metadata = controller_demonstrations(
        episodes, "train", maximum_budget=2
    )

    assert features.shape == (6, 9)
    assert actions.shape == (6,)
    assert {row["claim_id"] for row in metadata} == {"train-case"}
    assert {row["remaining_budget"] for row in metadata} == {0, 1, 2}
    train_sources = episodes[0]["sources"]
    probabilities = [
        constructed_nli_probabilities(source) for source in train_sources
    ]
    dependence = constructed_dependence(train_sources)
    for feature, record in zip(features, metadata):
        prefix = int(record["prefix"])
        index = list(range(prefix))
        expected = graph_state_features(
            probabilities[:prefix],
            dependence[np.ix_(index, index)],
            remaining_budget=int(record["remaining_budget"]),
        )
        np.testing.assert_array_equal(feature, expected)
    compact_features, compact_actions, compact_metadata = controller_demonstrations(
        episodes,
        "train",
        maximum_budget=2,
        include_metadata=False,
    )
    np.testing.assert_array_equal(compact_features, features)
    np.testing.assert_array_equal(compact_actions, actions)
    assert compact_metadata == []


def test_controller_summary_records_the_inference_boundary() -> None:
    patterns = (
        ("open", [("support", "a", 0.7)]),
        ("find", [("support", "a", 0.7), ("copy", "a", 0.7)]),
        ("verify", [("support", "a", 0.7), ("refute", "b", -0.7)]),
        ("search", [("neutral", "a", 0.0)]),
    )
    episodes = []
    for split in ("train", "dev"):
        for name, values in patterns:
            episodes.append(
                {
                    "claim_id": f"{split}-{name}",
                    "split": split,
                    "sources": [
                        _source(
                            f"{split}-{name}-{source_id}",
                            f"{split}-{origin}",
                            stance,
                        )
                        for source_id, origin, stance in values
                    ],
                }
            )
    _controller, summary = train_and_evaluate_controller(episodes, maximum_budget=2)

    assert summary["training_split"] == "train"
    assert summary["evaluation_split"] == "dev"
    assert summary["protocol"] == CONTROLLER_PROTOCOL
    assert summary["teacher_utility_terms"] == [
        "answer_sufficiency",
        "uncertainty",
        "independent_evidence_gain",
        "conflict_resolution",
        "acquisition_cost",
    ]
    assert "origin_id" in summary["construction_target_fields"]
    assert "origin_id" not in summary["deployed_feature_names"]
    assert set(summary["protected_fields_excluded_from_deployed_features"]).isdisjoint(
        summary["deployed_feature_names"]
    )
