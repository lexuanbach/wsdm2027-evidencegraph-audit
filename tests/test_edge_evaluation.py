from __future__ import annotations

import numpy as np

from evidencegraph.edge_evaluation import (
    clustered_metric_summary,
    dependence_examples,
    source_from_record,
)
from evidencegraph.dependence_features import pair_features


def _source(source_id: str, family: str, origin: str, text: str) -> dict:
    return {
        "source_id": source_id,
        "text": text,
        "family": family,
        "stance": 1.0,
        "stance_confidence": 0.95,
        "domain": f"{source_id}.example",
        "origin_id": origin,
        "cited_source_ids": [],
        "synthetic": False,
        "metadata": {},
    }


def test_dependence_examples_are_split_local_and_pair_aligned() -> None:
    episodes = [
        {
            "claim_id": "c1",
            "split": "train",
            "sources": [
                _source("a", "independent", "o1", "Apollo reached the Moon."),
                _source("b", "mirror", "o1", "Apollo reached the Moon."),
                _source("c", "independent", "o2", "A separate report."),
            ],
        },
        {
            "claim_id": "c2",
            "split": "test",
            "sources": [
                _source("d", "independent", "o3", "Different evidence."),
                _source("e", "mirror", "o3", "Different evidence."),
            ],
        },
    ]

    features, labels, metadata = dependence_examples(episodes, "train")

    assert features.shape == (3, 9)
    assert labels.tolist() == [1, 0, 0]
    assert {record["claim_id"] for record in metadata} == {"c1"}
    sources = [source_from_record(record) for record in episodes[0]["sources"]]
    expected = np.vstack(
        [
            pair_features(sources[0], sources[1]),
            pair_features(sources[0], sources[2]),
            pair_features(sources[1], sources[2]),
        ]
    )
    np.testing.assert_array_equal(features, expected)
    compact_features, compact_labels, compact_metadata = dependence_examples(
        episodes, "train", include_metadata=False
    )
    np.testing.assert_array_equal(compact_features, features)
    np.testing.assert_array_equal(compact_labels, labels)
    assert compact_metadata == []


def test_clustered_summary_reports_family_intervals() -> None:
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=int)
    probabilities = np.asarray([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
    families = ["independent", "mirror", "citation", "paraphrase"]
    metadata = [
        {
            "claim_id": f"c{index // 2}",
            "left_family": family,
            "right_family": families[(index + 1) % len(families)],
        }
        for index, family in enumerate(
            ["independent", "mirror", "citation", "paraphrase", "mirror", "citation"]
        )
    ]

    summary = clustered_metric_summary(
        labels, probabilities, metadata, n_resamples=100, seed=103
    )

    assert summary["groups"]["overall"]["accuracy"] == 1.0
    assert summary["groups"]["overall"]["accuracy_low"] == 1.0
    assert set(summary["groups"]) == {
        "overall",
        "independent",
        "mirror",
        "citation",
        "paraphrase",
    }
