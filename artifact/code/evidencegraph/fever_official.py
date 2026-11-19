"""Fail-closed adapter for the pinned official FEVER scorer."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
from types import ModuleType
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_evaluation import resolve_benchmark_attempts


FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
METRIC_NAMES = (
    "fever_score",
    "label_accuracy",
    "evidence_precision",
    "evidence_recall",
    "evidence_f1",
)


def parse_fever_source_id(source_id: str) -> list[str | int]:
    """Parse the final colon as the line separator; FEVER page IDs may contain colons."""
    page_id, separator, raw_line = str(source_id).rpartition(":")
    if not separator or not page_id:
        raise ValueError(f"invalid FEVER source ID: {source_id!r}")
    try:
        line_number = int(raw_line)
    except ValueError as error:
        raise ValueError(f"invalid FEVER line number: {source_id!r}") from error
    if line_number < 0:
        raise ValueError(f"negative FEVER line number: {source_id!r}")
    return [unicodedata.normalize("NFC", page_id), line_number]


def align_official_fever_records(
    actual: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align complete project predictions to every official FEVER record exactly once."""
    actual_index: dict[str, dict[str, Any]] = {}
    for record in actual:
        case_id = str(record["id"])
        if case_id in actual_index:
            raise ValueError(f"duplicate official FEVER ID: {case_id}")
        if str(record.get("label")) not in FEVER_LABELS or "evidence" not in record:
            raise ValueError(f"invalid official FEVER annotation: {case_id}")
        actual_index[case_id] = record

    fever_attempts = [record for record in predictions if record.get("dataset") == "FEVER"]
    resolved = resolve_benchmark_attempts(fever_attempts)
    prediction_index: dict[str, dict[str, Any]] = {}
    for record in resolved:
        case_id = str(record["case_id"])
        if case_id in prediction_index:
            raise ValueError(f"duplicate FEVER prediction: {case_id}")
        label = str(record.get("label"))
        if label not in FEVER_LABELS:
            raise ValueError(f"invalid FEVER prediction label for {case_id}: {label!r}")
        citations = record.get("cited_source_ids")
        if not isinstance(citations, list):
            raise ValueError(f"FEVER prediction lacks cited_source_ids: {case_id}")
        evidence = [parse_fever_source_id(source_id) for source_id in citations]
        if len(evidence) != len({(item[0], item[1]) for item in evidence}):
            raise ValueError(f"duplicate FEVER predicted evidence: {case_id}")
        prediction_index[case_id] = {
            "id": record["case_id"],
            "predicted_label": label,
            "predicted_evidence": evidence,
        }

    missing = sorted(set(actual_index) - set(prediction_index))
    extra = sorted(set(prediction_index) - set(actual_index))
    if missing or extra:
        raise ValueError(
            "FEVER predictions do not exactly cover the official release: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    ordered_ids = [str(record["id"]) for record in actual]
    return [prediction_index[case_id] for case_id in ordered_ids], [
        actual_index[case_id] for case_id in ordered_ids
    ]


def load_official_fever_module(
    scorer_root: Path,
) -> tuple[ModuleType, str]:
    """Load the pinned official scorer module and return its Git commit."""
    scorer_root = scorer_root.resolve()
    source = scorer_root / "src" / "fever" / "scorer.py"
    if not source.is_file() or not (scorer_root / ".git").exists():
        raise FileNotFoundError(
            f"official FEVER scorer checkout is missing or unpinned: {scorer_root}"
        )
    commit = subprocess.run(
        ["git", "-C", str(scorer_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = importlib.util.spec_from_file_location("evidencegraph_pinned_fever_scorer", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official FEVER scorer from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("fever_score", "evidence_macro_precision", "evidence_macro_recall")
    if any(not callable(getattr(module, name, None)) for name in required):
        raise ImportError(f"official FEVER scorer lacks required functions: {source}")
    return module, commit


def load_official_fever_score(
    scorer_root: Path,
) -> tuple[Callable[..., tuple[float, float, float, float, float]], str]:
    """Load fever_score directly from a pinned checkout and return its Git commit."""

    module, commit = load_official_fever_module(scorer_root)
    return module.fever_score, commit


def _score(
    score_fn: Callable[..., tuple[float, float, float, float, float]],
    predictions: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    max_evidence: int,
) -> dict[str, float]:
    # The official blind-evaluation path injects gold fields into predictions.
    # Deep copies keep the caller's immutable aligned records unchanged.
    values = score_fn(
        copy.deepcopy(predictions), copy.deepcopy(actual), max_evidence=max_evidence
    )
    if len(values) != len(METRIC_NAMES):
        raise ValueError("official FEVER scorer returned the wrong metric cardinality")
    return {name: float(value) for name, value in zip(METRIC_NAMES, values)}


def summarize_official_fever(
    predictions: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    score_fn: Callable[..., tuple[float, float, float, float, float]],
    *,
    max_evidence: int = 5,
    n_resamples: int = 1000,
    seed: int = 103,
) -> dict[str, Any]:
    """Run the exact scorer and claim bootstrap without changing scorer semantics."""
    if not predictions or len(predictions) != len(actual):
        raise ValueError("official FEVER summary requires aligned non-empty records")
    if max_evidence < 1 or n_resamples < 100:
        raise ValueError("max_evidence must be positive and at least 100 bootstraps are required")
    point = _score(score_fn, predictions, actual, max_evidence=max_evidence)
    rng = np.random.default_rng(seed)
    bootstrap = {name: [] for name in METRIC_NAMES}
    for _ in range(n_resamples):
        indices = rng.integers(0, len(predictions), size=len(predictions))
        sample_predictions = [predictions[int(index)] for index in indices]
        sample_actual = [actual[int(index)] for index in indices]
        values = _score(
            score_fn, sample_predictions, sample_actual, max_evidence=max_evidence
        )
        for name, value in values.items():
            bootstrap[name].append(value)
    output: dict[str, Any] = {
        "n_cases": len(predictions),
        "max_evidence": max_evidence,
        "bootstrap_resamples": n_resamples,
        **point,
    }
    for name, values in bootstrap.items():
        low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
        output[f"{name}_low"] = float(low)
        output[f"{name}_high"] = float(high)
    return output
