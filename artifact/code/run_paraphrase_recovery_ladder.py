"""Run the frozen primary paraphrase-recovery ladder with safe resumption.

This driver starts only after the complete v11 structural validation exists.  Every
subsequent tier is restricted to the current semantic failures, is audited before
validation, and is merged only after exact selection coverage is established.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from evidencegraph.dataset_builder import read_jsonl
from evidencegraph.deterministic_preordering import (
    COORDINATION_MODEL_ID,
    COORDINATION_PROMPT_VERSION,
    EXPANDED_MODEL_ID,
    EXPANDED_PROMPT_VERSION,
    MODEL_ID as DETERMINISTIC_MODEL_ID,
    PROMPT_VERSION as DETERMINISTIC_PROMPT_VERSION,
    PSEUDOCLEFT_MODEL_ID,
    PSEUDOCLEFT_PROMPT_VERSION,
)
from evidencegraph.paraphrase_audit import audit_generation_cache
from evidencegraph.paraphrase_job import PROMPT_PROFILES
from evidencegraph.paraphrase_structural_filter import STRUCTURE_PROTOCOL
from evidencegraph.paraphrase_validation_audit import audit_validation_tranche
from evidencegraph.semantic_validation import (
    PARAPHRASE_NLI_MODEL,
    PARAPHRASE_NLI_PROTOCOL,
    PARAPHRASE_NLI_REVISION,
)
from merge_paraphrase_validations import merge_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RecoveryStage:
    name: str
    kind: str
    model: str
    prompt_version: str
    prompt_profile: str | None = None
    batch_size: int | None = None
    deterministic_flag: str | None = None


STAGES = (
    RecoveryStage(
        "v12",
        "local",
        "qwen2.5:1.5b",
        PROMPT_PROFILES["lexically_anchored_v12"],
        prompt_profile="lexically_anchored_v12",
        batch_size=4,
    ),
    RecoveryStage(
        "v13",
        "local",
        "qwen2.5:3b-instruct",
        PROMPT_PROFILES["structure_checklist_v13"],
        prompt_profile="structure_checklist_v13",
        batch_size=2,
    ),
    RecoveryStage(
        "v16",
        "local",
        "qwen2.5:3b-instruct",
        PROMPT_PROFILES["token_preserving_preorder_v16"],
        prompt_profile="token_preserving_preorder_v16",
        batch_size=1,
    ),
    RecoveryStage(
        "v17",
        "deterministic",
        DETERMINISTIC_MODEL_ID,
        DETERMINISTIC_PROMPT_VERSION,
    ),
    RecoveryStage(
        "v18",
        "deterministic",
        COORDINATION_MODEL_ID,
        COORDINATION_PROMPT_VERSION,
        deterministic_flag="--include-coordination",
    ),
    RecoveryStage(
        "v19",
        "deterministic",
        PSEUDOCLEFT_MODEL_ID,
        PSEUDOCLEFT_PROMPT_VERSION,
        deterministic_flag="--pseudocleft-only",
    ),
    RecoveryStage(
        "v20",
        "deterministic",
        EXPANDED_MODEL_ID,
        EXPANDED_PROMPT_VERSION,
        deterministic_flag="--expanded-syntax",
    ),
)


@dataclass(frozen=True)
class StagePaths:
    failure_selection: Path
    eligible_selection: Path
    raw: Path
    generation_audit: Path
    nli_validation: Path
    structural_validation: Path
    validation_audit: Path
    cumulative_validation: Path
    cumulative_audit: Path


def stage_paths(output_dir: Path, stage: RecoveryStage) -> StagePaths:
    prefix = output_dir / stage.name
    eligible = (
        prefix.with_name(f"{stage.name}_eligible_selection.jsonl")
        if stage.kind == "deterministic"
        else prefix.with_name(f"{stage.name}_failure_selection.jsonl")
    )
    return StagePaths(
        failure_selection=prefix.with_name(f"{stage.name}_failure_selection.jsonl"),
        eligible_selection=eligible,
        raw=prefix.with_name(f"{stage.name}_raw.jsonl"),
        generation_audit=prefix.with_name(f"{stage.name}_generation_audit.json"),
        nli_validation=prefix.with_name(f"{stage.name}_nli_validated.jsonl"),
        structural_validation=prefix.with_name(f"{stage.name}_validated.jsonl"),
        validation_audit=prefix.with_name(f"{stage.name}_validation_audit.json"),
        cumulative_validation=prefix.with_name(
            f"primary_through_{stage.name}_validated.jsonl"
        ),
        cumulative_audit=prefix.with_name(
            f"primary_through_{stage.name}_validation_audit.json"
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_exact(path: Path, records: list[dict[str, Any]]) -> None:
    """Create an immutable JSONL artifact, or verify an existing exact match."""

    if path.exists():
        if read_jsonl(path) != records:
            raise ValueError(f"refusing to replace mismatched recovery artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_jsonl(path) != records:
                raise ValueError(
                    f"concurrent mismatched recovery artifact appeared: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exact(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise ValueError(f"refusing to replace mismatched recovery audit: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            with path.open(encoding="utf-8") as handle:
                if json.load(handle) != payload:
                    raise ValueError(
                        f"concurrent mismatched recovery audit appeared: {path}"
                    )
    finally:
        temporary.unlink(missing_ok=True)


def exact_failure_selection(
    selection: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return failures in frozen selection order after exact coverage checks."""

    selection_ids: list[str] = []
    seen_selection_ids: set[str] = set()
    for record in selection:
        claim_id = str(record["claim_id"])
        if claim_id in seen_selection_ids:
            raise ValueError(f"duplicate primary selection claim ID: {claim_id}")
        selection_ids.append(claim_id)
        seen_selection_ids.add(claim_id)
    statuses: dict[str, str] = {}
    for record in validation:
        claim_id = str(record["claim_id"])
        if claim_id in statuses:
            raise ValueError(f"duplicate validation claim ID: {claim_id}")
        status = str(record.get("status"))
        if status not in {"validated", "semantic_failure"}:
            raise ValueError(f"invalid validation status for {claim_id}: {status}")
        statuses[claim_id] = status
    expected = set(selection_ids)
    missing = expected - set(statuses)
    extra = set(statuses) - expected
    if missing or extra:
        raise ValueError(
            "recovery requires exact primary validation coverage: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return [
        record
        for record, claim_id in zip(selection, selection_ids)
        if statuses[claim_id] == "semantic_failure"
    ]


def validate_initial_structural_provenance(
    selection: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> None:
    exact_failure_selection(selection, validation)
    for record in validation:
        claim_id = str(record["claim_id"])
        if (
            record.get("nli_validator_model") != PARAPHRASE_NLI_MODEL
            or record.get("nli_validator_revision") != PARAPHRASE_NLI_REVISION
            or record.get("nli_validator_protocol") != PARAPHRASE_NLI_PROTOCOL
            or record.get("validator_revision") != PARAPHRASE_NLI_REVISION
            or record.get("validator_protocol") != PARAPHRASE_NLI_PROTOCOL
            or record.get("structural_gate_protocol") != STRUCTURE_PROTOCOL
            or not str(record.get("structural_parser_model", "")).strip()
            or not str(record.get("structural_parser_model_version", "")).strip()
            or not str(record.get("spacy_version", "")).strip()
            or float(record.get("threshold", -1.0)) != 0.80
        ):
            raise ValueError(
                f"initial validation does not use the frozen composite gate: {claim_id}"
            )


def _run(command: list[str], *, offline: bool = False) -> None:
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(CODE_DIR)
        if not prior_pythonpath
        else f"{CODE_DIR}{os.pathsep}{prior_pythonpath}"
    )
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps({"event": "run", "command": command}), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def _audit_generation(
    selection: Path, raw: Path, stage: RecoveryStage, output: Path
) -> dict[str, Any]:
    summary = audit_generation_cache(
        selection,
        raw,
        expected_model=stage.model,
        expected_prompt_version=stage.prompt_version,
        require_complete=True,
    )
    _write_json_exact(output, summary)
    return summary


def _complete_local_generation(
    stage: RecoveryStage,
    paths: StagePaths,
    *,
    max_rounds: int,
) -> dict[str, Any]:
    if max_rounds < 1:
        raise ValueError("max generation rounds must be positive")
    selection_count = len(read_jsonl(paths.eligible_selection))
    if selection_count == 0:
        _write_jsonl_exact(paths.raw, [])
        return _audit_generation(
            paths.eligible_selection, paths.raw, stage, paths.generation_audit
        )
    for _round in range(max_rounds):
        if paths.raw.exists():
            progress = audit_generation_cache(
                paths.eligible_selection,
                paths.raw,
                expected_model=stage.model,
                expected_prompt_version=stage.prompt_version,
                require_complete=False,
            )
            if progress["complete"]:
                break
        _run(
            [
                sys.executable,
                str(CODE_DIR / "run_paraphrases.py"),
                "--model",
                stage.model,
                "--batch-size",
                str(stage.batch_size),
                "--prompt-profile",
                str(stage.prompt_profile),
                "--selection",
                str(paths.eligible_selection),
                "--output",
                str(paths.raw),
            ]
        )
    return _audit_generation(
        paths.eligible_selection, paths.raw, stage, paths.generation_audit
    )


def _complete_deterministic_generation(
    stage: RecoveryStage, paths: StagePaths
) -> dict[str, Any]:
    if not paths.eligible_selection.exists() and not paths.raw.exists():
        command = [
            sys.executable,
            str(CODE_DIR / "run_deterministic_preordering.py"),
            "--selection",
            str(paths.failure_selection),
            "--eligible-selection",
            str(paths.eligible_selection),
            "--output",
            str(paths.raw),
            "--parser-model",
            "en_core_web_sm",
        ]
        if stage.deterministic_flag is not None:
            command.append(stage.deterministic_flag)
        _run(command)
    elif not paths.eligible_selection.exists() or not paths.raw.exists():
        raise ValueError(
            f"incomplete deterministic generation artifacts for {stage.name}"
        )
    eligible_ids = {
        str(record["claim_id"]) for record in read_jsonl(paths.eligible_selection)
    }
    failure_ids = {
        str(record["claim_id"]) for record in read_jsonl(paths.failure_selection)
    }
    if not eligible_ids.issubset(failure_ids):
        raise ValueError(f"{stage.name} eligible selection is not failure-only")
    return _audit_generation(
        paths.eligible_selection, paths.raw, stage, paths.generation_audit
    )


def _validate_exact_ids(selection_path: Path, validation_path: Path) -> None:
    selected = [str(record["claim_id"]) for record in read_jsonl(selection_path)]
    validated = [str(record["claim_id"]) for record in read_jsonl(validation_path)]
    if len(selected) != len(set(selected)) or len(validated) != len(set(validated)):
        raise ValueError("tier selection or validation contains duplicate claim IDs")
    if set(selected) != set(validated):
        raise ValueError(
            "tier validation does not exactly cover its eligible selection: "
            f"selected={len(selected)}, validated={len(validated)}"
        )


def _complete_validation(
    paths: StagePaths,
    *,
    device: str,
    record_batch_size: int,
    pair_batch_size: int,
) -> None:
    if not read_jsonl(paths.raw):
        _write_jsonl_exact(paths.nli_validation, [])
        _write_jsonl_exact(paths.structural_validation, [])
        _write_json_exact(
            paths.validation_audit,
            {
                "promotable": False,
                "reason": "no_structurally_eligible_records",
                "status": "skipped",
            },
        )
        return
    _run(
        [
            sys.executable,
            str(CODE_DIR / "validate_paraphrases.py"),
            "--raw",
            str(paths.raw),
            "--output",
            str(paths.nli_validation),
            "--model",
            PARAPHRASE_NLI_MODEL,
            "--revision",
            PARAPHRASE_NLI_REVISION,
            "--protocol",
            PARAPHRASE_NLI_PROTOCOL,
            "--device",
            device,
            "--record-batch-size",
            str(record_batch_size),
            "--pair-batch-size",
            str(pair_batch_size),
        ],
        offline=True,
    )
    _validate_exact_ids(paths.eligible_selection, paths.nli_validation)
    _run(
        [
            sys.executable,
            str(CODE_DIR / "apply_paraphrase_structural_gate.py"),
            "--validation",
            str(paths.nli_validation),
            "--output",
            str(paths.structural_validation),
            "--parser-model",
            "en_core_web_sm",
        ]
    )
    _validate_exact_ids(paths.eligible_selection, paths.structural_validation)
    audit = audit_validation_tranche(
        paths.eligible_selection,
        paths.structural_validation,
        project_root=PROJECT_ROOT,
    )
    if not audit["complete_selection"]:
        raise ValueError("tier validation audit is incomplete")
    _write_json_exact(paths.validation_audit, audit)


def _merge_and_audit(
    selection_path: Path,
    current_validation: Path,
    paths: StagePaths,
) -> tuple[Path, dict[str, Any]]:
    merged = merge_records([current_validation, paths.structural_validation])
    _write_jsonl_exact(paths.cumulative_validation, merged)
    selection = read_jsonl(selection_path)
    exact_failure_selection(selection, merged)
    audit = audit_validation_tranche(
        selection_path,
        paths.cumulative_validation,
        project_root=PROJECT_ROOT,
    )
    if not audit["complete_selection"]:
        raise ValueError("cumulative validation audit is incomplete")
    _write_json_exact(paths.cumulative_audit, audit)
    return paths.cumulative_validation, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
    )
    parser.add_argument(
        "--initial-validation",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/paraphrases/production_nfc_v11_qwen15_validated.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/paraphrases/production_nfc_full_recovery",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-generation-rounds", type=int, default=3)
    parser.add_argument("--record-batch-size", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--select-adaptive-reserves", action="store_true")
    parser.add_argument(
        "--fever",
        type=Path,
        default=PROJECT_ROOT / "data/raw/fever_train.jsonl",
    )
    args = parser.parse_args()

    selection = read_jsonl(args.selection)
    initial_validation = read_jsonl(args.initial_validation)
    validate_initial_structural_provenance(selection, initial_validation)
    current_validation = args.initial_validation
    stage_summaries: list[dict[str, Any]] = []

    for stage in STAGES:
        paths = stage_paths(args.output_dir, stage)
        failures = exact_failure_selection(selection, read_jsonl(current_validation))
        _write_jsonl_exact(paths.failure_selection, failures)
        if stage.kind == "local":
            _write_jsonl_exact(paths.eligible_selection, failures)
            generation = _complete_local_generation(
                stage,
                paths,
                max_rounds=args.max_generation_rounds,
            )
        else:
            generation = _complete_deterministic_generation(stage, paths)
        _complete_validation(
            paths,
            device=args.device,
            record_batch_size=args.record_batch_size,
            pair_batch_size=args.pair_batch_size,
        )
        current_validation, cumulative_audit = _merge_and_audit(
            args.selection, current_validation, paths
        )
        stage_summaries.append(
            {
                "stage": stage.name,
                "input_failures": len(failures),
                "eligible": int(generation["selection_count"]),
                "generated": int(generation["generated_count"]),
                "cumulative_passed": int(cumulative_audit["overall"]["passed"]),
                "cumulative_failed": int(cumulative_audit["overall"]["failed"]),
                "cumulative_validation": str(current_validation),
                "cumulative_validation_sha256": _sha256(current_validation),
            }
        )
        print(json.dumps(stage_summaries[-1], sort_keys=True), flush=True)

    final_rows = read_jsonl(current_validation)
    final_failures = exact_failure_selection(selection, final_rows)
    reserve_output = args.output_dir / "adaptive_reserve_selection.jsonl"
    reserve_audit = args.output_dir / "adaptive_reserve_selection_audit.json"
    if args.select_adaptive_reserves:
        if reserve_output.exists() != reserve_audit.exists():
            raise ValueError("adaptive reserve output is only partially present")
        if not reserve_output.exists():
            _run(
                [
                    sys.executable,
                    str(CODE_DIR / "select_adaptive_reserves.py"),
                    "--primary",
                    str(args.selection),
                    "--fever",
                    str(args.fever),
                    "--validation",
                    str(current_validation),
                    "--per-failure",
                    "2",
                    "--output",
                    str(reserve_output),
                    "--audit",
                    str(reserve_audit),
                ]
            )

    summary = {
        "schema_version": 1,
        "protocol": "primary-failure-only-paraphrase-recovery-ladder-v1",
        "selection": str(args.selection),
        "selection_sha256": _sha256(args.selection),
        "initial_validation": str(args.initial_validation),
        "initial_validation_sha256": _sha256(args.initial_validation),
        "final_validation": str(current_validation),
        "final_validation_sha256": _sha256(current_validation),
        "primary_count": len(selection),
        "passed": len(selection) - len(final_failures),
        "failed": len(final_failures),
        "stages": stage_summaries,
        "adaptive_reserves_selected": args.select_adaptive_reserves,
        "reserve_selection": str(reserve_output) if args.select_adaptive_reserves else None,
        "reserve_audit": str(reserve_audit) if args.select_adaptive_reserves else None,
    }
    _write_json_exact(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
