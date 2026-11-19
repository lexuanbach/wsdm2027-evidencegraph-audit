"""Raw-run completeness checks and bootstrap summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def validate_duplication_protocol(
    records: list[dict[str, Any]],
    *,
    expected_model: str,
    expected_protocol: str,
    expected_selection_seed: int,
) -> dict[str, Any]:
    """Verify that every controlled-intervention row follows the frozen design."""
    if not records or any(record.get("status") != "complete" for record in records):
        raise ValueError("duplication inputs contain incomplete records")
    for record in records:
        method = record.get("method")
        copy_count = int(record.get("copy_count", -1))
        if (
            record.get("model") != expected_model
            or record.get("protocol_version") != expected_protocol
            or int(record.get("selection_seed", -1)) != expected_selection_seed
            or record.get("confidence_mode") != "logprob"
            or record.get("gold_label") != "REFUTED"
            or int(record.get("true_support_origins", -1)) != 1
        ):
            raise ValueError("duplication input violates frozen run provenance")
        if method == "fixed_rag":
            expected_sources = 20
            expected_support_count = copy_count
        elif method == "evidencegraph_oracle":
            expected_sources = 1
            expected_support_count = 1
            if bool(record.get("false_claim_accepted")):
                raise ValueError("known-provenance gate accepted a one-origin false claim")
        else:
            raise ValueError(f"unexpected duplication method: {method}")
        if (
            int(record.get("n_sources_presented", -1)) != expected_sources
            or int(record.get("presented_support_documents", -1))
            != expected_support_count
            or int(record.get("aggregation_support_count", -1))
            != expected_support_count
        ):
            raise ValueError("duplication input violates matched-context accounting")
    return {
        "model": expected_model,
        "protocol_version": expected_protocol,
        "selection_seed": expected_selection_seed,
        "confidence_mode": "logprob",
        "record_count": len(records),
    }


def bootstrap_mean(
    values: list[float], n_resamples: int = 1000, seed: int = 103
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = np.mean(array[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(array)), float(low), float(high)


def summarize_duplication(
    records: list[dict[str, Any]],
    *,
    expected_claims: int,
    methods: tuple[str, ...],
    copy_counts: tuple[int, ...],
    n_resamples: int = 1000,
    seed: int = 103,
) -> dict[str, Any]:
    complete = [record for record in records if record.get("status") == "complete"]
    expected = expected_claims * len(methods) * len(copy_counts)
    if len(complete) != expected:
        raise ValueError(f"incomplete duplication run: {len(complete)} of {expected}")
    keys = [
        (record["claim_id"], record["method"], int(record["copy_count"]))
        for record in complete
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate completed case keys")
    claim_ids = {record["claim_id"] for record in complete}
    if len(claim_ids) != expected_claims:
        raise ValueError("completed run has the wrong number of unique claims")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in complete:
        grouped[(record["method"], int(record["copy_count"]))].append(record)
    output: dict[str, Any] = {
        "copy_count": list(copy_counts),
        "n_claims": expected_claims,
        "bootstrap_resamples": n_resamples,
        "methods": {},
    }
    for method in methods:
        method_record = {
            "confidence_mean": [],
            "confidence_low": [],
            "confidence_high": [],
            "false_acceptance_mean": [],
            "false_acceptance_low": [],
            "false_acceptance_high": [],
        }
        for copy_count in copy_counts:
            cells = grouped[(method, copy_count)]
            if len(cells) != expected_claims:
                raise ValueError(f"incomplete cell {method}/{copy_count}")
            for key, field in (
                ("confidence", "confidence_in_false_claim"),
                ("false_acceptance", "false_claim_accepted"),
            ):
                mean, low, high = bootstrap_mean(
                    [float(record[field]) for record in cells],
                    n_resamples=n_resamples,
                    seed=seed + copy_count,
                )
                method_record[f"{key}_mean"].append(mean)
                method_record[f"{key}_low"].append(low)
                method_record[f"{key}_high"].append(high)
        output["methods"][method] = method_record
    return output


def duplication_holds(
    summary: dict[str, Any],
    baseline: str = "fixed_rag",
    evidencegraph: str = "evidencegraph_oracle",
    minimum_baseline_increase: float = 0.05,
    maximum_graph_increase: float = 0.05,
) -> dict[str, float | bool]:
    baseline_values = summary["methods"][baseline]["confidence_mean"]
    graph_values = summary["methods"][evidencegraph]["confidence_mean"]
    baseline_increase = float(baseline_values[-1] - baseline_values[0])
    graph_increase = float(graph_values[-1] - graph_values[0])
    return {
        "baseline_increase": baseline_increase,
        "evidencegraph_increase": graph_increase,
        "baseline_inflates": baseline_increase >= minimum_baseline_increase,
        "evidencegraph_stable": graph_increase <= maximum_graph_increase,
        "headline_holds": baseline_increase >= minimum_baseline_increase
        and graph_increase <= maximum_graph_increase,
    }


def summarize_duplication_multiseed(
    records: list[dict[str, Any]],
    *,
    expected_claims: int,
    expected_seeds: tuple[int, ...],
    methods: tuple[str, ...],
    copy_counts: tuple[int, ...],
    n_resamples: int = 1000,
    seed: int = 103,
) -> dict[str, Any]:
    """Average fixed-seed repetitions per claim, then cluster-bootstrap claims."""
    complete = [record for record in records if record.get("status") == "complete"]
    expected = expected_claims * len(expected_seeds) * len(methods) * len(copy_counts)
    if len(complete) != expected:
        raise ValueError(f"incomplete multiseed duplication run: {len(complete)} of {expected}")
    keys = [
        (
            record["claim_id"],
            int(record["seed"]),
            record["method"],
            int(record["copy_count"]),
        )
        for record in complete
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate multiseed completed case keys")
    claim_ids = sorted({str(record["claim_id"]) for record in complete})
    if len(claim_ids) != expected_claims:
        raise ValueError("multiseed run has the wrong number of claims")
    lookup = {
        (str(record["claim_id"]), int(record["seed"]), record["method"], int(record["copy_count"])): record
        for record in complete
    }
    output: dict[str, Any] = {
        "copy_count": list(copy_counts),
        "n_claims": expected_claims,
        "n_seeds": len(expected_seeds),
        "seeds": list(expected_seeds),
        "bootstrap_resamples": n_resamples,
        "methods": {},
    }
    for method in methods:
        method_record = {
            "confidence_mean": [],
            "confidence_low": [],
            "confidence_high": [],
            "false_acceptance_mean": [],
            "false_acceptance_low": [],
            "false_acceptance_high": [],
        }
        for copy_count in copy_counts:
            for key, field in (
                ("confidence", "confidence_in_false_claim"),
                ("false_acceptance", "false_claim_accepted"),
            ):
                claim_means = [
                    float(
                        np.mean(
                            [
                                float(lookup[(claim_id, run_seed, method, copy_count)][field])
                                for run_seed in expected_seeds
                            ]
                        )
                    )
                    for claim_id in claim_ids
                ]
                mean, low, high = bootstrap_mean(
                    claim_means, n_resamples=n_resamples, seed=seed + copy_count
                )
                method_record[f"{key}_mean"].append(mean)
                method_record[f"{key}_low"].append(low)
                method_record[f"{key}_high"].append(high)
        output["methods"][method] = method_record
    return output
