from __future__ import annotations

from evidencegraph.summarize import (
    duplication_holds,
    summarize_duplication,
    summarize_duplication_multiseed,
    validate_duplication_protocol,
)


def test_duplication_summary_and_diagnostic() -> None:
    records = []
    for claim_id in ("a", "b"):
        for method in ("fixed_rag", "evidencegraph_oracle"):
            for copies in (1, 20):
                confidence = 0.2 + (0.4 if method == "fixed_rag" and copies == 20 else 0.0)
                records.append(
                    {
                        "claim_id": claim_id,
                        "method": method,
                        "copy_count": copies,
                        "confidence_in_false_claim": confidence,
                        "false_claim_accepted": confidence >= 0.5,
                        "status": "complete",
                    }
                )
    summary = summarize_duplication(
        records,
        expected_claims=2,
        methods=("fixed_rag", "evidencegraph_oracle"),
        copy_counts=(1, 20),
        n_resamples=100,
    )
    diagnostic = duplication_holds(summary)
    assert diagnostic["headline_holds"]


def test_multiseed_summary_bootstraps_claim_means() -> None:
    records = []
    for claim_id in ("a", "b"):
        for run_seed in (1, 2):
            for method in ("fixed_rag", "evidencegraph_oracle"):
                for copies in (1, 20):
                    confidence = 0.1 + (
                        0.4 if method == "fixed_rag" and copies == 20 else 0.0
                    )
                    records.append(
                        {
                            "claim_id": claim_id,
                            "seed": run_seed,
                            "method": method,
                            "copy_count": copies,
                            "confidence_in_false_claim": confidence,
                            "false_claim_accepted": confidence >= 0.5,
                            "status": "complete",
                        }
                    )
    summary = summarize_duplication_multiseed(
        records,
        expected_claims=2,
        expected_seeds=(1, 2),
        methods=("fixed_rag", "evidencegraph_oracle"),
        copy_counts=(1, 20),
        n_resamples=100,
    )
    assert summary["n_seeds"] == 2
    assert duplication_holds(summary)["headline_holds"]


def test_duplication_protocol_validation_checks_matched_accounting() -> None:
    record = {
        "claim_id": "a",
        "seed": 103,
        "selection_seed": 103,
        "method": "fixed_rag",
        "copy_count": 5,
        "model": "model",
        "protocol_version": "protocol",
        "confidence_mode": "logprob",
        "gold_label": "REFUTED",
        "true_support_origins": 1,
        "n_sources_presented": 20,
        "presented_support_documents": 5,
        "aggregation_support_count": 5,
        "false_claim_accepted": False,
        "status": "complete",
    }
    metadata = validate_duplication_protocol(
        [record],
        expected_model="model",
        expected_protocol="protocol",
        expected_selection_seed=103,
    )
    assert metadata["record_count"] == 1

    record["n_sources_presented"] = 5
    try:
        validate_duplication_protocol(
            [record],
            expected_model="model",
            expected_protocol="protocol",
            expected_selection_seed=103,
        )
    except ValueError as error:
        assert "matched-context" in str(error)
    else:
        raise AssertionError("a non-matched RAG context must be rejected")
