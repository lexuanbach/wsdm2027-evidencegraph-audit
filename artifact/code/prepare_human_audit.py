"""Create the blinded rater packet and coordinator-only source key."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from evidencegraph.dataset_builder import read_jsonl, write_jsonl
from evidencegraph.human_audit_package import (
    AUDIT_PACKET_PROTOCOL,
    build_audit_package,
)
from evidencegraph.result_registry import source_descriptor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=PROJECT_ROOT / "data/audit/rater_packet/tasks.jsonl",
    )
    parser.add_argument(
        "--source-key",
        type=Path,
        default=PROJECT_ROOT / "data/audit/coordinator/source_key.jsonl",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "data/audit/AUDIT_PROTOCOL.md",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/rater_packet/AUDIT_PROTOCOL.md",
    )
    parser.add_argument(
        "--rater-app",
        type=Path,
        default=PROJECT_ROOT / "artifact/code/run_human_audit_rater.py",
    )
    parser.add_argument(
        "--rater-app-output",
        type=Path,
        default=PROJECT_ROOT / "data/audit/rater_packet/run_human_audit_rater.py",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/audit/coordinator/audit_packet_manifest.json",
    )
    args = parser.parse_args()
    test = [record for record in read_jsonl(args.dataset) if record["split"] == "test"]
    if len(test) != 3_000:
        raise ValueError(f"human audit requires exactly 3,000 test claims, found {len(test)}")
    tasks, source_key = build_audit_package(test)
    write_jsonl(tasks, args.tasks)
    write_jsonl(source_key, args.source_key)
    for source, output in (
        (args.protocol, args.protocol_output),
        (args.rater_app, args.rater_app_output),
    ):
        if source.resolve() == output.resolve():
            raise ValueError("rater bundle source and output paths must differ")
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.partial")
        shutil.copyfile(source, partial)
        os.replace(partial, output)
    manifest = {
        "status": "prepared",
        "protocol": AUDIT_PACKET_PROTOCOL,
        "n_claims": len(tasks),
        "n_sources": len(source_key),
        "rater_distribution": [
            source_descriptor(args.tasks, PROJECT_ROOT),
            source_descriptor(args.protocol_output, PROJECT_ROOT),
            source_descriptor(args.rater_app_output, PROJECT_ROOT),
        ],
        "coordinator_only": [source_descriptor(args.source_key, PROJECT_ROOT)],
        "dataset": source_descriptor(args.dataset, PROJECT_ROOT),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
