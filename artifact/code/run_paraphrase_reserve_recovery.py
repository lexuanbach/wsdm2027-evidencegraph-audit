"""Recover validated reserves, finalize EvidenceGraph-20K, and materialize it.

The first reserve manifest is sized from complete post-v20 primary failures
and checksum-verified February-2017 Wikidata coverage failures.
If two candidates per required replacement do not yield enough validated
reserves in a stratum, later rounds select a new disjoint window sized from the
observed shortfall.  No threshold, stratum, or episode-count fallback exists.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from evidencegraph.dataset_builder import (
    choose_reserve_claims,
    read_jsonl,
    reserve_candidate_strata,
)
from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from evidencegraph.paraphrase_job import PROMPT_PROFILES
from evidencegraph.reserve_replacement import finalize_selection, selection_stratum
from evidencegraph.result_registry import source_descriptor
from evidencegraph.wikidata_snapshot import (
    derive_snapshot_subset,
    load_provenance_index,
    provenance_usable_page_ids,
)
from merge_paraphrase_validations import merge_records
from run_paraphrase_recovery_ladder import (
    CODE_DIR,
    PROJECT_ROOT,
    RecoveryStage,
    _complete_local_generation,
    _complete_validation,
    _run,
    _sha256,
    _write_json_exact,
    _write_jsonl_exact,
    exact_failure_selection,
    stage_paths,
    validate_initial_structural_provenance,
)


V11_STAGE = RecoveryStage(
    "v11",
    "local",
    "qwen2.5:1.5b",
    PROMPT_PROFILES["faithful_v11"],
    prompt_profile="faithful_v11",
    batch_size=4,
)


def _index_validation(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        claim_id = str(record["claim_id"])
        if claim_id in indexed:
            raise ValueError(f"duplicate validation claim ID: {claim_id}")
        if record.get("status") not in {"validated", "semantic_failure"}:
            raise ValueError(f"invalid validation status for {claim_id}")
        indexed[claim_id] = record
    return indexed


def reserve_shortfalls(
    primary: list[dict[str, Any]],
    reserves: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    eligible_claim_ids: set[str] | None = None,
) -> dict[tuple[str, str, str], int]:
    """Return required additional validated reserves by frozen stratum."""

    validation = _index_validation(validation_records)
    primary_ids = {str(record["claim_id"]) for record in primary}
    reserve_ids = {str(record["claim_id"]) for record in reserves}
    if len(primary_ids) != len(primary) or len(reserve_ids) != len(reserves):
        raise ValueError("primary or reserve selections contain duplicate claim IDs")
    if primary_ids & reserve_ids:
        raise ValueError("primary and reserve claim IDs overlap")
    required_ids = primary_ids | reserve_ids
    missing = required_ids - set(validation)
    extra = set(validation) - required_ids
    if missing or extra:
        raise ValueError(
            "reserve shortfall requires exact combined validation coverage: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    demand = Counter(
        selection_stratum(record)
        for record in primary
        if (
            validation[str(record["claim_id"])]["status"] == "semantic_failure"
            or (
                eligible_claim_ids is not None
                and str(record["claim_id"]) not in eligible_claim_ids
            )
        )
    )
    supply = Counter(
        selection_stratum(record)
        for record in reserves
        if (
            validation[str(record["claim_id"])]["status"] == "validated"
            and (
                eligible_claim_ids is None
                or str(record["claim_id"]) in eligible_claim_ids
            )
        )
    )
    return {
        stratum: demand_count - supply[stratum]
        for stratum, demand_count in sorted(demand.items())
        if supply[stratum] < demand_count
    }


def select_additional_reserves(
    fever_path: Path,
    primary: list[dict[str, Any]],
    prior_reserves: list[dict[str, Any]],
    shortfalls: dict[tuple[str, str, str], int],
    *,
    per_shortfall: int = 2,
    seed: int = 103,
    eligible_claim_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if per_shortfall < 1:
        raise ValueError("per-shortfall reserve count must be positive")
    prior_ids = {str(record["claim_id"]) for record in prior_reserves}
    if len(prior_ids) != len(prior_reserves):
        raise ValueError("prior reserve selections contain duplicate claim IDs")
    strata = reserve_candidate_strata(fever_path, primary, seed=seed)
    disjoint = {
        stratum: [
            record
            for record in records
            if str(record["claim_id"]) not in prior_ids
            and (
                eligible_claim_ids is None
                or str(record["claim_id"]) in eligible_claim_ids
            )
        ]
        for stratum, records in strata.items()
    }
    targets = {
        stratum: count * per_shortfall for stratum, count in shortfalls.items()
    }
    return choose_reserve_claims(disjoint, targets, seed=seed)


def provenance_eligible_claim_ids(
    records: list[dict[str, Any]], usable_page_ids: set[str]
) -> set[str]:
    """Return claims whose complete evidence set has usable snapshot provenance."""

    eligible: set[str] = set()
    seen: set[str] = set()
    for record in records:
        claim_id = str(record["claim_id"])
        if claim_id in seen:
            raise ValueError(f"duplicate selection claim ID: {claim_id}")
        seen.add(claim_id)
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"selection claim {claim_id} has no evidence")
        if any(
            not isinstance(reference, (list, tuple)) or len(reference) != 2
            for reference in evidence
        ):
            raise ValueError(f"selection claim {claim_id} has malformed evidence")
        page_ids = {str(reference[0]) for reference in evidence}
        if page_ids <= usable_page_ids:
            eligible.add(claim_id)
    return eligible


def replacement_demand(
    primary: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    *,
    eligible_claim_ids: set[str],
) -> Counter[tuple[str, str, str]]:
    """Count the union of semantic and provenance failures by frozen stratum."""

    validation = _index_validation(validation_records)
    primary_ids = {str(record["claim_id"]) for record in primary}
    if len(primary_ids) != len(primary) or set(validation) != primary_ids:
        raise ValueError("replacement demand requires exact primary validation")
    return Counter(
        selection_stratum(record)
        for record in primary
        if (
            validation[str(record["claim_id"])]["status"] == "semantic_failure"
            or str(record["claim_id"]) not in eligible_claim_ids
        )
    )


def select_initial_coverage_aware_reserves(
    reserve_pool: list[dict[str, Any]],
    demand: Counter[tuple[str, str, str]],
    *,
    eligible_claim_ids: set[str],
    per_failure: int = 2,
    seed: int = 103,
) -> list[dict[str, Any]]:
    """Select only snapshot-covered candidates for the first recovery round."""

    if per_failure < 1:
        raise ValueError("per-failure reserve count must be positive")
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in reserve_pool:
        claim_id = str(record["claim_id"])
        if claim_id not in eligible_claim_ids:
            continue
        strata.setdefault(selection_stratum(record), []).append(record)
    targets = {stratum: count * per_failure for stratum, count in demand.items()}
    return choose_reserve_claims(strata, targets, seed=seed)


def _resolved_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def freeze_combined_validation(
    records: list[dict[str, Any]],
    output: Path = FINAL_COMBINED_VALIDATION,
) -> Path:
    """Write or verify the stable validation handoff used by release tools."""

    _write_jsonl_exact(output, records)
    return output


def _run_reserve_round(
    selection_path: Path,
    round_dir: Path,
    *,
    device: str,
    max_generation_rounds: int,
    record_batch_size: int,
    pair_batch_size: int,
) -> Path:
    records = read_jsonl(selection_path)
    if not records:
        raise ValueError("a reserve recovery round cannot be empty")
    v11_paths = stage_paths(round_dir, V11_STAGE)
    _write_jsonl_exact(v11_paths.failure_selection, records)
    _complete_local_generation(
        V11_STAGE,
        v11_paths,
        max_rounds=max_generation_rounds,
    )
    _complete_validation(
        v11_paths,
        device=device,
        record_batch_size=record_batch_size,
        pair_batch_size=pair_batch_size,
    )
    validate_initial_structural_provenance(
        records, read_jsonl(v11_paths.structural_validation)
    )
    recovery_dir = round_dir / "recovery"
    _run(
        [
            sys.executable,
            str(CODE_DIR / "run_paraphrase_recovery_ladder.py"),
            "--selection",
            str(selection_path),
            "--initial-validation",
            str(v11_paths.structural_validation),
            "--output-dir",
            str(recovery_dir),
            "--device",
            device,
            "--max-generation-rounds",
            str(max_generation_rounds),
            "--record-batch-size",
            str(record_batch_size),
            "--pair-batch-size",
            str(pair_batch_size),
        ]
    )
    with (recovery_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    final_validation = _resolved_path(str(summary["final_validation"]))
    exact_failure_selection(records, read_jsonl(final_validation))
    return final_validation


def _verify_materialization(
    dataset_path: Path,
    stats_path: Path,
    *,
    selection_path: Path,
    validation_path: Path,
    wikidata_targets: Path,
    wikidata_terms: Path,
    wikidata_audit: Path,
) -> dict[str, Any]:
    if not dataset_path.exists() or not stats_path.exists():
        raise ValueError("materialized dataset or statistics are missing")
    with stats_path.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    selection = read_jsonl(selection_path)
    expected_hidden_best = sum(
        str(record.get("corruption")) == "hidden_best_evidence"
        for record in selection
    )
    if (
        int(stats.get("episodes", -1)) != 20_000
        or stats.get("protocol")
        != "evidencegraph-20k-v5-wikidata-budget-separated-decoy"
        or int(stats.get("hidden_best_intervention_episodes", -1))
        != expected_hidden_best
    ):
        raise ValueError(
            "materialized dataset violates the current construction protocol"
        )
    if len(read_jsonl(dataset_path)) != 20_000:
        raise ValueError("materialized dataset row count is not 20,000")
    expected_sources = [
        source_descriptor(path, PROJECT_ROOT)
        for path in (
            selection_path,
            validation_path,
            wikidata_targets,
            wikidata_terms,
            wikidata_audit,
        )
    ]
    if stats.get("construction_sources") != expected_sources:
        raise ValueError("materialized dataset sources do not match final inputs")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
    )
    parser.add_argument(
        "--primary-validation",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/paraphrases/production_nfc_full_recovery"
        / "primary_through_v20_validated.jsonl",
    )
    parser.add_argument(
        "--provenance-reserves",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evidencegraph20k_provenance_reserve.jsonl",
        help="Frozen reserve pool included in the checksum-bound snapshot extraction.",
    )
    parser.add_argument(
        "--fever",
        type=Path,
        default=PROJECT_ROOT / "data/raw/fever_train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/paraphrases/production_nfc_reserve_recovery",
    )
    parser.add_argument("--max-reserve-rounds", type=int, default=5)
    parser.add_argument("--per-shortfall", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-generation-rounds", type=int, default=3)
    parser.add_argument("--record-batch-size", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument(
        "--final-selection",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evidencegraph20k_final_selection.jsonl",
    )
    parser.add_argument(
        "--replacement-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_replacements.jsonl",
    )
    parser.add_argument(
        "--final-selection-stats",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evidencegraph20k_final_selection_stats.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_stats.json",
    )
    args = parser.parse_args()
    if args.max_reserve_rounds < 1:
        parser.error("--max-reserve-rounds must be positive")

    primary = read_jsonl(args.primary)
    primary_validation = read_jsonl(args.primary_validation)
    primary_failures = exact_failure_selection(primary, primary_validation)
    superset_targets = (
        PROJECT_ROOT / "data/processed/wikidata20170220_target_entities.jsonl"
    )
    superset_terms = (
        PROJECT_ROOT / "data/processed/wikidata20170220_reference_terms.jsonl"
    )
    superset_audit = (
        PROJECT_ROOT / "artifact/runs/wikidata20170220_snapshot_audit.json"
    )
    usable_page_ids = provenance_usable_page_ids(
        superset_targets, superset_terms, superset_audit
    )
    provenance_reserves = read_jsonl(args.provenance_reserves)
    eligible_claim_ids = provenance_eligible_claim_ids(
        primary + provenance_reserves, usable_page_ids
    )
    demand = replacement_demand(
        primary,
        primary_validation,
        eligible_claim_ids=eligible_claim_ids,
    )
    all_reserves: list[dict[str, Any]] = []
    validation_paths = [args.primary_validation]
    round_summaries: list[dict[str, Any]] = []

    if demand:
        initial_records = select_initial_coverage_aware_reserves(
            provenance_reserves,
            demand,
            eligible_claim_ids=eligible_claim_ids,
            per_failure=args.per_shortfall,
        )
        current_selection_path = (
            args.output_dir / "round_00" / "coverage_aware_reserve_selection.jsonl"
        )
        _write_jsonl_exact(current_selection_path, initial_records)
        for round_index in range(args.max_reserve_rounds):
            round_records = read_jsonl(current_selection_path)
            round_ids = [str(record["claim_id"]) for record in round_records]
            if not round_records or len(round_ids) != len(set(round_ids)):
                raise ValueError("reserve recovery round is empty or contains duplicates")
            prior_ids = {str(record["claim_id"]) for record in all_reserves}
            primary_ids = {str(record["claim_id"]) for record in primary}
            overlap = (prior_ids | primary_ids) & set(round_ids)
            if overlap:
                raise ValueError("reserve recovery overlaps a prior or primary selection")
            all_reserves.extend(round_records)
            round_dir = args.output_dir / f"round_{round_index:02d}"
            final_round_validation = _run_reserve_round(
                current_selection_path,
                round_dir,
                device=args.device,
                max_generation_rounds=args.max_generation_rounds,
                record_batch_size=args.record_batch_size,
                pair_batch_size=args.pair_batch_size,
            )
            validation_paths.append(final_round_validation)
            combined = merge_records(validation_paths)
            combined_path = (
                args.output_dir
                / f"combined_validation_through_round_{round_index:02d}.jsonl"
            )
            _write_jsonl_exact(combined_path, combined)
            shortfalls = reserve_shortfalls(
                primary,
                all_reserves,
                combined,
                eligible_claim_ids=eligible_claim_ids,
            )
            round_summary = {
                "round": round_index,
                "selection": str(current_selection_path),
                "selection_sha256": _sha256(current_selection_path),
                "selected": len(round_records),
                "combined_validation": str(combined_path),
                "combined_validation_sha256": _sha256(combined_path),
                "remaining_replacements": sum(shortfalls.values()),
                "shortfalls": {
                    "/".join(stratum): count
                    for stratum, count in sorted(shortfalls.items())
                },
            }
            round_summaries.append(round_summary)
            _write_json_exact(round_dir / "reserve_round_summary.json", round_summary)
            if not shortfalls:
                break
            if round_index + 1 == args.max_reserve_rounds:
                raise RuntimeError(
                    "validated reserve capacity remains insufficient after the "
                    f"maximum rounds: {shortfalls}"
                )
            next_records = select_additional_reserves(
                args.fever,
                primary,
                all_reserves,
                shortfalls,
                per_shortfall=args.per_shortfall,
                eligible_claim_ids=eligible_claim_ids,
            )
            current_selection_path = (
                args.output_dir
                / f"round_{round_index + 1:02d}"
                / "adaptive_reserve_selection.jsonl"
            )
            _write_jsonl_exact(current_selection_path, next_records)
    else:
        combined = primary_validation
        combined_path = args.primary_validation

    # Downstream release tooling must not have to infer which adaptive round
    # happened to satisfy capacity.  Freeze the exact merged validation under
    # one canonical path only after the reserve loop has reached zero shortfall.
    combined_path = freeze_combined_validation(combined)

    final, replacements, final_stats = finalize_selection(
        primary,
        all_reserves,
        combined,
        eligible_claim_ids=eligible_claim_ids,
    )
    _write_jsonl_exact(args.final_selection, final)
    _write_jsonl_exact(args.replacement_manifest, replacements)
    _write_json_exact(args.final_selection_stats, final_stats)

    wikidata_targets = FINAL_WIKIDATA_TARGETS
    wikidata_terms = FINAL_WIKIDATA_TERMS
    wikidata_audit = FINAL_WIKIDATA_AUDIT
    required_pages = {
        str(page_id)
        for record in final
        for page_id, _line_number in record["evidence"]
    }
    try:
        load_provenance_index(
            wikidata_targets,
            wikidata_terms,
            wikidata_audit,
            required_page_ids=required_pages,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        reused_superset = False
        if all(path.is_file() for path in (superset_targets, superset_terms, superset_audit)):
            try:
                derive_snapshot_subset(
                    args.final_selection,
                    superset_targets,
                    superset_terms,
                    superset_audit,
                    wikidata_targets,
                    wikidata_terms,
                    wikidata_audit,
                )
                reused_superset = True
            except (KeyError, TypeError, ValueError):
                reused_superset = False
        if not reused_superset:
            _run(
                [
                    sys.executable,
                    str(CODE_DIR / "extract_wikidata_snapshot.py"),
                    "--manifest",
                    str(args.final_selection),
                    "--targets",
                    str(wikidata_targets),
                    "--terms",
                    str(wikidata_terms),
                    "--audit",
                    str(wikidata_audit),
                ]
            )
        load_provenance_index(
            wikidata_targets,
            wikidata_terms,
            wikidata_audit,
            required_page_ids=required_pages,
        )

    if args.dataset.exists() != args.dataset_stats.exists():
        raise ValueError("materialized dataset is only partially present")
    if not args.dataset.exists():
        _run(
            [
                sys.executable,
                str(CODE_DIR / "materialize_dataset.py"),
                "--selection",
                str(args.final_selection),
                "--validation",
                str(combined_path),
                "--output",
                str(args.dataset),
                "--stats",
                str(args.dataset_stats),
                "--wikidata-targets",
                str(wikidata_targets),
                "--wikidata-terms",
                str(wikidata_terms),
                "--wikidata-audit",
                str(wikidata_audit),
                "--expected-count",
                "20000",
            ]
        )
    dataset_stats = _verify_materialization(
        args.dataset,
        args.dataset_stats,
        selection_path=args.final_selection,
        validation_path=combined_path,
        wikidata_targets=wikidata_targets,
        wikidata_terms=wikidata_terms,
        wikidata_audit=wikidata_audit,
    )
    _run(
        [
            sys.executable,
            str(CODE_DIR / "prepare_human_audit.py"),
            "--dataset",
            str(args.dataset),
        ]
    )
    audit_packet_manifest = (
        PROJECT_ROOT / "data/audit/coordinator/audit_packet_manifest.json"
    )
    with audit_packet_manifest.open(encoding="utf-8") as handle:
        audit_packet = json.load(handle)
    if (
        audit_packet.get("status") != "prepared"
        or audit_packet.get("n_claims") != 3_000
        or audit_packet.get("dataset")
        != source_descriptor(args.dataset, PROJECT_ROOT)
    ):
        raise ValueError("prepared human-audit packet violates the frozen dataset contract")
    summary = {
        "schema_version": 1,
        "protocol": "adaptive-provenance-covered-reserve-recovery-v2",
        "primary": source_descriptor(args.primary, PROJECT_ROOT),
        "primary_validation": source_descriptor(args.primary_validation, PROJECT_ROOT),
        "primary_failures": len(primary_failures),
        "primary_provenance_failures": sum(
            str(record["claim_id"]) not in eligible_claim_ids for record in primary
        ),
        "usable_snapshot_page_ids": len(usable_page_ids),
        "provenance_reserve_pool": source_descriptor(
            args.provenance_reserves, PROJECT_ROOT
        ),
        "reserve_rounds": round_summaries,
        "reserves_attempted": len(all_reserves),
        "replacements": len(replacements),
        "final_selection": source_descriptor(args.final_selection, PROJECT_ROOT),
        "combined_validation": source_descriptor(combined_path, PROJECT_ROOT),
        "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
        "dataset_stats": dataset_stats,
        "audit_packet_manifest": source_descriptor(
            audit_packet_manifest, PROJECT_ROOT
        ),
        "wikidata_audit": source_descriptor(wikidata_audit, PROJECT_ROOT),
    }
    _write_json_exact(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
