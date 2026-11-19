"""Claim-level retrieval coverage with explicit gold-evidence denominators."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .bootstrap import multinomial_weight_batches


_RETRIEVAL_ATTEMPT_STATUSES = frozenset({"complete", "retrieval_failure"})
_SELECTION_ATTEMPT_STATUSES = frozenset({"complete", "selection_failure"})


def resolve_retrieval_attempts(
    records: list[dict[str, Any]], *, require_complete: bool = True
) -> list[dict[str, Any]]:
    """Resolve append-only retrieval retries without discarding their audit trail."""
    if not records:
        if require_complete:
            raise ValueError("retrieval attempt resolution requires records")
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        status = str(record.get("status", ""))
        if status not in _RETRIEVAL_ATTEMPT_STATUSES:
            raise ValueError(f"unknown retrieval attempt status: {status!r}")
        key = (str(record.get("dataset", "")), str(record.get("case_id", "")))
        if not all(key):
            raise ValueError("retrieval attempt lacks dataset or case ID")
        groups.setdefault(key, []).append(record)
    resolved = []
    for key, attempts in groups.items():
        identities = {
            (
                str(record.get("query", "")),
                str(record.get("retrieval_protocol", "")),
                json.dumps(
                    record.get("retrieval_config"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for record in attempts
        }
        if len(identities) != 1 or not all(next(iter(identities))):
            raise ValueError(f"retrieval attempts mix run identity for {key}")
        complete = [record for record in attempts if record["status"] == "complete"]
        if len(complete) > 1:
            raise ValueError(f"retrieval case {key} has duplicate completions")
        if not complete:
            if require_complete:
                raise ValueError(f"retrieval case {key} has no completed attempt")
            continue
        completion_index = attempts.index(complete[0])
        if any(
            record["status"] != "complete"
            for record in attempts[completion_index + 1 :]
        ):
            raise ValueError(f"retrieval case {key} has failures after completion")
        resolved.append(complete[0])
    return resolved


def resolve_selection_attempts(
    records: list[dict[str, Any]], *, require_complete: bool = True
) -> list[dict[str, Any]]:
    """Resolve append-only graph-selection retries under one exact configuration."""
    if not records:
        if require_complete:
            raise ValueError("selection attempt resolution requires records")
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        status = str(record.get("status", ""))
        if status not in _SELECTION_ATTEMPT_STATUSES:
            raise ValueError(f"unknown selection attempt status: {status!r}")
        key = (str(record.get("dataset", "")), str(record.get("case_id", "")))
        if not all(key):
            raise ValueError("selection attempt lacks dataset or case ID")
        groups.setdefault(key, []).append(record)
    resolved = []
    for key, attempts in groups.items():
        identities = {
            (
                str(record.get("query", "")),
                str(record.get("selection_method", "")),
                json.dumps(
                    record.get("selection_config"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for record in attempts
        }
        if len(identities) != 1 or not all(next(iter(identities))):
            raise ValueError(f"selection attempts mix run identity for {key}")
        complete = [record for record in attempts if record["status"] == "complete"]
        if len(complete) > 1:
            raise ValueError(f"selection case {key} has duplicate completions")
        if not complete:
            if require_complete:
                raise ValueError(f"selection case {key} has no completed attempt")
            continue
        completion_index = attempts.index(complete[0])
        if any(
            record["status"] != "complete"
            for record in attempts[completion_index + 1 :]
        ):
            raise ValueError(f"selection case {key} has failures after completion")
        resolved.append(complete[0])
    return resolved


def retrieval_run_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = resolve_retrieval_attempts(records)
    datasets = {str(record.get("dataset", "")) for record in records}
    protocols = {str(record.get("retrieval_protocol", "")) for record in records}
    configs = {
        json.dumps(record.get("retrieval_config"), sort_keys=True, separators=(",", ":"))
        for record in records
    }
    if len(datasets) != 1 or len(protocols) != 1 or len(configs) != 1:
        raise ValueError("retrieval records mix dataset, protocol, or configuration")
    dataset, protocol = next(iter(datasets)), next(iter(protocols))
    if not dataset or not protocol or next(iter(configs)) == "null":
        raise ValueError("retrieval run metadata is incomplete")
    return {
        "dataset": dataset,
        "protocol": protocol,
        "configuration": json.loads(next(iter(configs))),
        "n_retrievals": len(records),
    }


def retrieval_case_records(
    cases: list[dict[str, Any]], retrievals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    retrievals = resolve_retrieval_attempts(retrievals)
    retrieval_index = {}
    for record in retrievals:
        case_id = str(record["case_id"])
        if case_id in retrieval_index:
            raise ValueError(f"duplicate retrieval case ID: {case_id}")
        retrieval_index[case_id] = record
    output = []
    case_ids = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in case_ids:
            raise ValueError(f"duplicate retrieval benchmark case ID: {case_id}")
        case_ids.add(case_id)
        if case_id not in retrieval_index:
            raise ValueError(f"missing retrieval for case {case_id}")
        record = retrieval_index[case_id]
        if case["dataset"] == "AVeriTeC":
            ranked = [str(result["url"]) for result in record["results"]]
        else:
            ranked = [str(result["source_id"]) for result in record["results"]]
        retrieved = set(ranked)
        gold = set(map(str, case.get("gold_evidence", [])))
        groups = [set(map(str, group)) for group in case.get("metadata", {}).get("evidence_groups", [])]
        if not groups and gold:
            groups = [gold]
        positions = [ranked.index(identifier) + 1 for identifier in gold if identifier in retrieved]
        output.append(
            {
                "dataset": str(case["dataset"]),
                "case_id": case_id,
                "gold_items": len(gold),
                "retrieved_items": len(ranked),
                "item_recall": len(gold & retrieved) / len(gold) if gold else None,
                "complete_group_hit": int(any(group <= retrieved for group in groups))
                if groups
                else None,
                "reciprocal_rank": 1.0 / min(positions) if positions else (0.0 if gold else None),
            }
        )
    extra = sorted(set(retrieval_index) - case_ids)
    if extra:
        raise ValueError(f"retrievals contain {len(extra)} extra cases")
    return output


def summarize_retrieval_cases(
    records: list[dict[str, Any]], *, n_resamples: int = 1000, seed: int = 103
) -> dict[str, Any]:
    if not records or n_resamples < 100:
        raise ValueError("retrieval summary requires records and 100 bootstraps")
    output = {}
    for group_index, dataset in enumerate(sorted({record["dataset"] for record in records})):
        group = [record for record in records if record["dataset"] == dataset]
        eligible = [record for record in group if record["gold_items"] > 0]
        if not eligible:
            raise ValueError(f"no gold-evidence retrieval denominator for {dataset}")
        values = np.asarray(
            [
                [
                    float(record["item_recall"]),
                    float(record["complete_group_hit"]),
                    float(record["reciprocal_rank"]),
                ]
                for record in eligible
            ],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed + group_index)
        # Explicit einsum avoids an Accelerate/BLAS mixed-shape matmul failure
        # observed on Apple Silicon for the 1000 x 210 bootstrap matrix.
        bootstrap = np.concatenate(
            [
                np.einsum(
                    "ij,jk->ik",
                    weights.astype(np.float64),
                    values,
                    optimize=False,
                )
                / len(eligible)
                for weights in multinomial_weight_batches(
                    rng, len(eligible), n_resamples
                )
            ],
            axis=0,
        )
        names = ("item_recall", "complete_group_recall", "mrr")
        summary: dict[str, Any] = {
            "n_cases": len(group),
            "n_cases_with_gold": len(eligible),
            "n_cases_without_gold": len(group) - len(eligible),
        }
        for metric_index, name in enumerate(names):
            low, high = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
            summary[name] = float(np.mean(values[:, metric_index]))
            summary[f"{name}_low"] = float(low)
            summary[f"{name}_high"] = float(high)
        output[dataset] = summary
    return {"bootstrap_resamples": n_resamples, "datasets": output}


def compare_retrieval_cases(
    candidate_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    *,
    n_resamples: int = 10_000,
    seed: int = 103,
) -> dict[str, Any]:
    """Return paired claim-bootstrap differences for two retrieval runs."""
    if not candidate_records or not baseline_records or n_resamples < 100:
        raise ValueError("paired retrieval comparison requires records and 100 bootstraps")

    def index(records: list[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
        output: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            key = (str(record.get("dataset", "")), str(record.get("case_id", "")))
            if not all(key) or key in output:
                raise ValueError(f"{name} retrieval cases have missing or duplicate identities")
            output[key] = record
        return output

    candidate = index(candidate_records, "candidate")
    baseline = index(baseline_records, "baseline")
    if set(candidate) != set(baseline):
        raise ValueError("paired retrieval comparison requires identical case identities")

    metric_fields = {
        "complete_group_recall": "complete_group_hit",
        "item_recall": "item_recall",
        "mrr": "reciprocal_rank",
    }
    output: dict[str, Any] = {}
    datasets = sorted({dataset for dataset, _case_id in candidate})
    for dataset_index, dataset in enumerate(datasets):
        keys = sorted(key for key in candidate if key[0] == dataset)
        eligible = []
        for key in keys:
            candidate_record = candidate[key]
            baseline_record = baseline[key]
            candidate_gold = int(candidate_record.get("gold_items", -1))
            baseline_gold = int(baseline_record.get("gold_items", -1))
            if candidate_gold != baseline_gold:
                raise ValueError(f"paired retrieval case {key} has mismatched gold counts")
            if candidate_gold > 0:
                eligible.append(key)
        if not eligible:
            raise ValueError(f"no paired gold-evidence denominator for {dataset}")

        differences = np.empty((len(eligible), len(metric_fields)), dtype=np.float64)
        for row_index, key in enumerate(eligible):
            for metric_index, field in enumerate(metric_fields.values()):
                candidate_value = float(candidate[key][field])
                baseline_value = float(baseline[key][field])
                if (
                    not np.isfinite(candidate_value)
                    or not np.isfinite(baseline_value)
                    or not 0.0 <= candidate_value <= 1.0
                    or not 0.0 <= baseline_value <= 1.0
                ):
                    raise ValueError(f"paired retrieval case {key} has invalid {field}")
                differences[row_index, metric_index] = candidate_value - baseline_value

        rng = np.random.default_rng(seed + dataset_index)
        bootstrap = np.concatenate(
            [
                np.einsum(
                    "ij,jk->ik",
                    weights.astype(np.float64),
                    differences,
                    optimize=False,
                )
                / len(eligible)
                for weights in multinomial_weight_batches(
                    rng, len(eligible), n_resamples
                )
            ],
            axis=0,
        )
        summary: dict[str, Any] = {
            "n_cases": len(keys),
            "n_cases_with_gold": len(eligible),
            "n_cases_without_gold": len(keys) - len(eligible),
        }
        for metric_index, name in enumerate(metric_fields):
            low, high = np.quantile(bootstrap[:, metric_index], [0.025, 0.975])
            summary[f"{name}_difference"] = float(
                np.mean(differences[:, metric_index])
            )
            summary[f"{name}_difference_low"] = float(low)
            summary[f"{name}_difference_high"] = float(high)
        output[dataset] = summary
    return {
        "bootstrap_resamples": n_resamples,
        "seed": seed,
        "datasets": output,
    }
