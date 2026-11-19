from __future__ import annotations

import pytest

from evidencegraph.nli_evaluation import (
    SCORING_PROTOCOL,
    gold_nli_label,
    nli_input_sha256,
    nli_run_metadata,
    summarize_nli_records,
    validate_nli_dataset,
    validate_nli_resume,
)


def test_gold_nli_label_maps_stance_sign() -> None:
    assert gold_nli_label(1.0) == "support"
    assert gold_nli_label(0.0) == "neutral"
    assert gold_nli_label(-1.0) == "refute"


def test_nli_summary_is_family_stratified_and_clustered() -> None:
    records = []
    families = ("independent", "mirror", "citation", "paraphrase")
    labels = ("support", "neutral", "refute")
    for claim_index in range(6):
        for family_index, family in enumerate(families):
            label = labels[(claim_index + family_index) % 3]
            probabilities = {name: 0.05 for name in labels}
            probabilities[label] = 0.90
            records.append(
                {
                    "claim_id": f"c{claim_index}",
                    "source_id": f"c{claim_index}-{family}",
                    "family": family,
                    "gold_label": label,
                    "probabilities": probabilities,
                    "model": "tiny-nli",
                    "model_revision": "abc",
                    "model_parameters": 10,
                    "scoring_protocol": SCORING_PROTOCOL,
                    "dataset_sha256": "dataset-sha",
                    "input_sha256": nli_input_sha256(
                        f"claim-{claim_index}", f"source-{family}"
                    ),
                    "split": "test",
                    "status": "complete",
                }
            )

    summary = summarize_nli_records(records, n_resamples=100, seed=103)

    assert summary["groups"]["overall"]["accuracy"] == 1.0
    assert summary["groups"]["mirror"]["macro_f1"] == 1.0
    assert summary["groups"]["paraphrase"]["accuracy_low"] == 1.0


def test_nli_resume_rejects_mixed_checkpoint_or_unexpected_key() -> None:
    expected = {
        ("c1", "s1"): (
            "independent",
            "support",
            nli_input_sha256("claim", "source"),
        )
    }
    record = {
        "claim_id": "c1",
        "source_id": "s1",
        "split": "test",
        "family": "independent",
        "gold_label": "support",
        "model": "model",
        "model_revision": "revision",
        "scoring_protocol": SCORING_PROTOCOL,
        "dataset_sha256": "dataset-sha",
        "input_sha256": nli_input_sha256("claim", "source"),
        "status": "complete",
    }
    assert validate_nli_resume(
        [record],
        expected,
        split="test",
        model="model",
        revision="revision",
        dataset_sha256="dataset-sha",
    ) == {("c1", "s1")}

    record["model_revision"] = "changed"
    with pytest.raises(ValueError, match="frozen run configuration"):
        validate_nli_resume(
            [record],
            expected,
            split="test",
            model="model",
            revision="revision",
            dataset_sha256="dataset-sha",
        )

    record["model_revision"] = "revision"
    record["input_sha256"] = nli_input_sha256("claim", "changed source")
    with pytest.raises(ValueError, match="frozen run configuration"):
        validate_nli_resume(
            [record],
            expected,
            split="test",
            model="model",
            revision="revision",
            dataset_sha256="dataset-sha",
        )


def test_nli_metadata_allows_dev_and_test_from_one_checkpoint() -> None:
    records = []
    for split in ("dev", "test"):
        records.append(
            {
                "model": "model",
                "model_revision": "revision",
                "model_parameters": 10,
                "scoring_protocol": SCORING_PROTOCOL,
                "dataset_sha256": "dataset-sha",
                "split": split,
                "status": "complete",
            }
        )
    metadata = nli_run_metadata(records)
    assert metadata["splits"] == ["dev", "test"]
    assert metadata["records"] == 2
    assert metadata["dataset_sha256"] == "dataset-sha"


def test_nli_dataset_binding_rejects_a_different_materialization() -> None:
    records = [
        {
            "model": "model",
            "model_revision": "revision",
            "model_parameters": 10,
            "scoring_protocol": SCORING_PROTOCOL,
            "dataset_sha256": "dataset-sha",
            "split": "test",
            "status": "complete",
        }
    ]
    assert validate_nli_dataset(records, "dataset-sha")["dataset_sha256"] == (
        "dataset-sha"
    )
    with pytest.raises(ValueError, match="different dataset checksum"):
        validate_nli_dataset(records, "changed-dataset-sha")
