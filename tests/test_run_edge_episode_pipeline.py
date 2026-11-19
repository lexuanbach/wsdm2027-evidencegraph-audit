from __future__ import annotations

from pathlib import Path

import json

import pytest

from evidencegraph.result_registry import source_descriptor
from run_edge_episode_pipeline import (
    PROJECT_ROOT,
    _dependence_complete,
    dependence_denominators,
    nli_run_specs,
)


def test_nli_run_specs_cover_both_models_and_evaluation_splits(tmp_path: Path) -> None:
    specs = nli_run_specs(tmp_path)

    assert {(record["tag"], record["split"]) for record in specs} == {
        ("deberta_large", "dev"),
        ("deberta_large", "test"),
        ("minilm", "dev"),
        ("minilm", "test"),
    }
    assert all(record["output"].parent == tmp_path for record in specs)


def test_pipeline_runs_coordinated_fabrication_boundary() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "artifact/code/run_edge_episode_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "run_coordinated_fabrication_stress.py" in script
    assert "coordinated_fabrication_summary" in script


def test_dependence_resume_requires_exact_corpus_pair_denominators(
    tmp_path: Path,
) -> None:
    episodes = []
    for split, count in (("train", 14_000), ("dev", 3_000), ("test", 3_000)):
        episodes.extend(
            {
                "claim_id": f"{split}-{index}",
                "split": split,
                "sources": [{"source_id": "a"}, {"source_id": "b"}],
            }
            for index in range(count)
        )
    split_claims, split_pairs = dependence_denominators(episodes)
    assert split_claims == {"train": 14_000, "dev": 3_000, "test": 3_000}
    assert split_pairs == split_claims

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n" * 6_000, encoding="utf-8")
    models = []
    for seed in (103, 104, 105):
        model = tmp_path / f"model-{seed}.joblib"
        model.write_text("model", encoding="utf-8")
        models.append(source_descriptor(model, PROJECT_ROOT))
    summary_path = tmp_path / "summary.json"
    summary = {
        "protocol": "dependence-logistic-calibrated-v3-token-cosine",
        "feature_names": [
            "token_embedding_cosine",
            "token_jaccard",
            "char_4gram_jaccard",
            "shared_rare_token_rate",
            "same_domain",
            "explicit_citation",
            "same_author",
            "known_temporal_order",
            "log_time_gap_days",
        ],
        "seeds": [103, 104, 105],
        "train_pairs": 14_000,
        "dev_pairs": 3_000,
        "test_pairs": 3_000,
        "inputs": {
            "split_claims": split_claims,
            "dataset": source_descriptor(dataset, PROJECT_ROOT),
            "predictions": source_descriptor(predictions, PROJECT_ROOT),
            "models": models,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert _dependence_complete(
        summary_path, dataset, split_claims, split_pairs
    )

    summary["test_pairs"] -= 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen contract"):
        _dependence_complete(summary_path, dataset, split_claims, split_pairs)
