import json
from pathlib import Path

import pytest

from build_dataset_registry import (
    PROJECT_ROOT,
    dataset_registry,
    parse_args,
    verify_construction_sources,
    verify_selection_statistics,
)
from evidencegraph.dataset_builder import selection_statistics
from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from evidencegraph.result_registry import source_descriptor


def test_dataset_registry_requires_full_two_rater_test_audit() -> None:
    selection = {
        "episodes": 20_000,
        "split_counts": {"train": 14_000, "dev": 3_000, "test": 3_000},
        "topic_leakage": {"train_dev": 0, "train_test": 0, "dev_test": 0},
        "topic_counts": {"train": 2500, "dev": 500, "test": 500},
        "label_counts": {
            "train:SUPPORTS": 7000,
            "train:REFUTES": 7000,
            "dev:SUPPORTS": 1500,
            "dev:REFUTES": 1500,
            "test:SUPPORTS": 1500,
            "test:REFUTES": 1500,
        },
        "corruption_counts": {
            "duplication": 5000,
            "paraphrase_flood": 5000,
            "support_refute_mix": 5000,
            "hidden_best_evidence": 5000,
        },
        "replacement_count": 12,
    }
    dataset = {
        "protocol": "evidencegraph-20k-v5-wikidata-budget-separated-decoy",
        "episodes": 20_000,
        "wikidata_unique_entities": 4_000,
        "sources": 350_000,
        "minimum_sources_per_episode": 5,
        "maximum_sources_per_episode": 30,
        "split_counts": {"train": 14_000, "dev": 3_000, "test": 3_000},
        "hidden_best_intervention_episodes": 5000,
        "family_counts": {
            "independent": 200_000,
            "mirror": 50_000,
            "citation": 50_000,
            "paraphrase": 50_000,
        },
    }
    audit = {
        "status": "complete",
        "protocol": "two-independent-raters-v2-blinded-aliases",
        "target_kappa": 0.8,
        "target_met": True,
        "metrics": {
            "n_claims": 3000,
            "n_sources": 52_000,
            "n_claim_ratings": 6000,
            "rater_count": 2,
            "verdict_raw_agreement": 0.9,
            "verdict_kappa": 0.85,
            "family_raw_agreement": 0.88,
            "family_kappa": 0.82,
        },
    }
    table = dataset_registry(selection, dataset, audit, replacement_rows=12)
    assert len(table["rows"]) == 17
    assert table["caption"] == (
        "EvidenceGraph-20K composition, leakage checks, and independent audit."
    )
    assert table["rows"][-1]["test"] == "88.0 / 0.820"

    audit["target_met"] = False
    audit["status"] = "below_target"
    with pytest.raises(ValueError, match="frozen construction contract"):
        dataset_registry(selection, dataset, audit, replacement_rows=12)


def test_dataset_registry_verifies_materialization_sources(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    validation = tmp_path / "validation.jsonl"
    selection.write_text(json.dumps({"claim_id": "c1"}) + "\n", encoding="utf-8")
    validation.write_text(json.dumps({"claim_id": "c1"}) + "\n", encoding="utf-8")
    dataset = {
        "construction_sources": [
            source_descriptor(selection, PROJECT_ROOT),
            source_descriptor(validation, PROJECT_ROOT),
        ]
    }

    verify_construction_sources(dataset, selection, validation)
    dataset["construction_sources"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor mismatch"):
        verify_construction_sources(dataset, selection, validation)


def test_dataset_registry_recomputes_final_selection_statistics() -> None:
    records = [
        {
            "claim_id": "c1",
            "split": "test",
            "fever_label": "SUPPORTS",
            "corruption": "duplication",
            "topic_id": "topic-1",
        }
    ]
    summary = {
        **selection_statistics(records),
        "replacement_count": 0,
        "original_episode_count": 1,
    }

    verify_selection_statistics(summary, records, replacement_rows=0)
    summary["corruption_counts"] = {"duplication": 2}
    with pytest.raises(ValueError, match="statistics mismatch"):
        verify_selection_statistics(summary, records, replacement_rows=0)


def test_dataset_registry_defaults_target_final_release_artifacts() -> None:
    args = parse_args([])
    assert args.validation == FINAL_COMBINED_VALIDATION
    assert args.wikidata_targets == FINAL_WIKIDATA_TARGETS
    assert args.wikidata_terms == FINAL_WIKIDATA_TERMS
    assert args.wikidata_audit == FINAL_WIKIDATA_AUDIT
