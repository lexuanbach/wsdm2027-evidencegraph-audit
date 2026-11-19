"""Build the measured EvidenceGraph-20K composition and human-audit table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidencegraph.audit import validate_audit
from evidencegraph.audit import KAPPA_TARGET, meets_kappa_target
from evidencegraph.dataset_builder import read_jsonl, selection_statistics
from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from evidencegraph.human_audit_package import validate_audit_package
from evidencegraph.result_registry import (
    source_descriptor,
    verified_source_path,
    write_complete_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}"


def verify_construction_sources(
    dataset: dict[str, Any], *source_paths: Path
) -> None:
    embedded = dataset.get("construction_sources")
    actual = [source_descriptor(path, PROJECT_ROOT) for path in source_paths]
    if not isinstance(embedded, list) or len(embedded) != len(actual):
        raise ValueError("dataset stats are missing construction source descriptors")
    for expected, observed in zip(actual, embedded):
        if observed != expected:
            raise ValueError("dataset construction source descriptor mismatch")


def verify_selection_statistics(
    summary: dict[str, Any], records: list[dict[str, Any]], replacement_rows: int
) -> None:
    recomputed = selection_statistics(records)
    for key, expected in recomputed.items():
        if summary.get(key) != expected:
            raise ValueError(f"final selection statistics mismatch: {key}")
    if (
        int(summary.get("replacement_count", -1)) != replacement_rows
        or int(summary.get("original_episode_count", -1)) != len(records)
    ):
        raise ValueError("final selection replacement statistics mismatch")


def verify_audit_summary(
    audit: dict[str, Any], dataset_path: Path
) -> list[Path]:
    inputs = audit.get("inputs")
    expected = {
        "dataset",
        "tasks",
        "source_key",
        "claim_ratings",
        "source_ratings",
        "audit_protocol",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected:
        raise ValueError("human audit summary input descriptors are incomplete")
    paths = {
        name: verified_source_path(descriptor, PROJECT_ROOT)
        for name, descriptor in inputs.items()
    }
    if paths["dataset"].resolve() != dataset_path.resolve():
        raise ValueError("human audit summary uses a different dataset")
    test = [
        record for record in read_jsonl(paths["dataset"]) if record["split"] == "test"
    ]
    if len(test) != 3_000:
        raise ValueError("human audit summary does not cover 3,000 test claims")
    claim_ids, source_ids = validate_audit_package(
        test,
        read_jsonl(paths["tasks"]),
        read_jsonl(paths["source_key"]),
    )
    recomputed = validate_audit(
        paths["claim_ratings"],
        paths["source_ratings"],
        claim_ids,
        source_ids,
    )
    if audit.get("metrics") != recomputed:
        raise ValueError("human audit summary metrics do not recompute")
    return list(paths.values())


def _unique_paths(paths: list[Path]) -> list[Path]:
    output = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            output.append(path)
            seen.add(resolved)
    return output


def dataset_registry(
    selection: dict[str, Any],
    dataset: dict[str, Any],
    audit: dict[str, Any],
    replacement_rows: int,
) -> dict[str, Any]:
    if (
        selection.get("episodes") != 20_000
        or selection.get("split_counts") != {"train": 14_000, "dev": 3_000, "test": 3_000}
        or any(selection.get("topic_leakage", {}).values())
        or dataset.get("episodes") != 20_000
        or dataset.get("protocol")
        != "evidencegraph-20k-v5-wikidata-budget-separated-decoy"
        or int(dataset.get("wikidata_unique_entities", 0)) < 1
        or dataset.get("minimum_sources_per_episode") != 5
        or dataset.get("maximum_sources_per_episode") != 30
        or dataset.get("split_counts") != {"train": 14_000, "dev": 3_000, "test": 3_000}
        or dataset.get("hidden_best_intervention_episodes")
        != selection.get("corruption_counts", {}).get("hidden_best_evidence")
        or set(dataset.get("family_counts", {}))
        != {"independent", "mirror", "citation", "paraphrase"}
        or audit.get("status") != "complete"
        or audit.get("protocol") != "two-independent-raters-v2-blinded-aliases"
        or audit.get("target_kappa") != KAPPA_TARGET
        or audit.get("target_met") is not True
        or not meets_kappa_target(audit.get("metrics", {}))
    ):
        raise ValueError("dataset or audit summary violates the frozen construction contract")
    metrics = audit["metrics"]
    if (
        metrics.get("n_claims") != 3000
        or metrics.get("n_claim_ratings") != 6000
        or metrics.get("rater_count") != 2
        or replacement_rows != int(selection.get("replacement_count", 0))
    ):
        raise ValueError("dataset audit or replacement denominator is incomplete")
    labels = selection["label_counts"]
    rows = [
        {"measure": "Episodes", "overall": "20,000", "train": "14,000", "dev": "3,000", "test": "3,000"},
        {
            "measure": "Supported / refuted",
            "overall": "10,000 / 10,000",
            "train": f"{labels['train:SUPPORTS']:,} / {labels['train:REFUTES']:,}",
            "dev": f"{labels['dev:SUPPORTS']:,} / {labels['dev:REFUTES']:,}",
            "test": f"{labels['test:SUPPORTS']:,} / {labels['test:REFUTES']:,}",
        },
        {
            "measure": "Topics",
            "overall": f"{sum(selection['topic_counts'].values()):,}",
            "train": f"{selection['topic_counts']['train']:,}",
            "dev": f"{selection['topic_counts']['dev']:,}",
            "test": f"{selection['topic_counts']['test']:,}",
        },
        {
            "measure": "Sources (range/episode)",
            "overall": f"{dataset['sources']:,} (5--30)",
            "train": "--",
            "dev": "--",
            "test": "--",
        },
        {
            "measure": "Date-aligned Wikidata entities",
            "overall": f"{dataset['wikidata_unique_entities']:,}",
            "train": "--",
            "dev": "--",
            "test": "--",
        },
    ]
    for family in ("independent", "mirror", "citation", "paraphrase"):
        rows.append(
            {
                "measure": f"Sources: {family}",
                "overall": f"{dataset['family_counts'][family]:,}",
                "train": "--",
                "dev": "--",
                "test": "--",
            }
        )
    for key, display in (
        ("duplication", "Episodes: duplication"),
        ("paraphrase_flood", "Episodes: paraphrase flood"),
        ("support_refute_mix", "Episodes: support/refute mix"),
        ("hidden_best_evidence", "Episodes: hidden best evidence"),
    ):
        rows.append(
            {
                "measure": display,
                "overall": f"{selection['corruption_counts'][key]:,}",
                "train": "--",
                "dev": "--",
                "test": "--",
            }
        )
    rows.extend(
        [
            {
                "measure": "Cross-split topic overlap",
                "overall": "0",
                "train": "0",
                "dev": "0",
                "test": "0",
            },
            {
                "measure": "Same-stratum replacements",
                "overall": f"{replacement_rows:,}",
                "train": "--",
                "dev": "--",
                "test": "--",
            },
            {
                "measure": "Two-rater verdict agreement / $\\kappa$",
                "overall": "--",
                "train": "--",
                "dev": "--",
                "test": (
                    f"{_pct(metrics['verdict_raw_agreement'])} / "
                    f"{metrics['verdict_kappa']:.3f}"
                ),
            },
            {
                "measure": "Two-rater family agreement / $\\kappa$",
                "overall": "--",
                "train": "--",
                "dev": "--",
                "test": (
                    f"{_pct(metrics['family_raw_agreement'])} / "
                    f"{metrics['family_kappa']:.3f}"
                ),
            },
        ]
    )
    return {
        "caption": (
            "EvidenceGraph-20K composition, leakage checks, and independent audit."
        ),
        "columns": [
            {"key": "measure", "header": "Measure"},
            {"key": "overall", "header": "Overall"},
            {"key": "train", "header": "Train"},
            {"key": "dev", "header": "Dev"},
            {"key": "test", "header": "Test"},
        ],
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_final_selection_stats.json",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_stats.json",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_final_selection.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=FINAL_COMBINED_VALIDATION,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--wikidata-targets",
        type=Path,
        default=FINAL_WIKIDATA_TARGETS,
    )
    parser.add_argument(
        "--wikidata-terms",
        type=Path,
        default=FINAL_WIKIDATA_TERMS,
    )
    parser.add_argument(
        "--wikidata-audit",
        type=Path,
        default=FINAL_WIKIDATA_AUDIT,
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=PROJECT_ROOT / "data/audit/audit_summary.json",
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_replacements.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifact/results/registries",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    with args.replacements.open(encoding="utf-8") as handle:
        replacement_rows = sum(1 for line in handle if line.strip())
    selection_stats = _read(args.selection_stats)
    dataset_stats = _read(args.dataset_stats)
    audit_summary = _read(args.audit_summary)
    verify_selection_statistics(
        selection_stats, read_jsonl(args.selection), replacement_rows
    )
    verify_construction_sources(
        dataset_stats,
        args.selection,
        args.validation,
        args.wikidata_targets,
        args.wikidata_terms,
        args.wikidata_audit,
    )
    audit_sources = verify_audit_summary(audit_summary, args.dataset)
    registry = dataset_registry(
        selection_stats,
        dataset_stats,
        audit_summary,
        replacement_rows,
    )
    output = write_complete_registry(
        "dataset",
        registry,
        _unique_paths([
            args.selection_stats,
            args.selection,
            args.validation,
            args.wikidata_targets,
            args.wikidata_terms,
            args.wikidata_audit,
            args.dataset_stats,
            args.dataset,
            args.audit_summary,
            args.replacements,
            *audit_sources,
        ]),
        args.output_dir,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps({"registry": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
