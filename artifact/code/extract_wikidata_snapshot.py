"""Build the date-aligned Wikidata subset used by EvidenceGraph-20K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.wikidata_snapshot import build_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump",
        type=Path,
        default=PROJECT_ROOT / "data/raw/wikidata-20170220-all.json.bz2",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="Repeat for selection and reserve manifests.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "data/processed/wikidata20170220_target_entities.jsonl",
    )
    parser.add_argument(
        "--terms",
        type=Path,
        default=PROJECT_ROOT / "data/processed/wikidata20170220_reference_terms.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "artifact/runs/wikidata20170220_snapshot_audit.json",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    manifests = args.manifest or [
        PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
        PROJECT_ROOT
        / "data/processed/evidencegraph20k_provenance_reserve.jsonl",
    ]
    audit = build_snapshot(
        args.dump,
        manifests,
        args.targets,
        args.terms,
        args.audit,
    )
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "complete" and not args.allow_missing:
        raise SystemExit(
            "Wikidata snapshot is incomplete; inspect the audit and repair missing pages"
        )


if __name__ == "__main__":
    main()
