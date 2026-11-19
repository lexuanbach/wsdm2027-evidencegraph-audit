"""Fail-closed check that every submission display still matches its sources.

``build_subset_tables.py`` writes ``artifact/results/subset_tables_manifest.json``
binding each rendered LaTeX table to the run summaries it was produced from, by
sha256 and byte length.  Nothing in the test suite re-checked those bindings, so
a hand-edited summary or a stale table could ship silently.  This script is that
check, and it is the last step of ``rebuild_submission.sh``.

Exit status is 0 only when every binding matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from evidencegraph.provenance import sha256


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "artifact/results/subset_tables_manifest.json"


def binding_failures(manifest_path: Path, project_root: Path) -> list[str]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest:
        return [f"{manifest_path}: manifest is empty"]
    failures: list[str] = []
    for display, entry in sorted(manifest.items()):
        descriptors = list(entry.get("sources", []))
        table = entry.get("table")
        if table is None:
            failures.append(f"{display}: manifest entry has no rendered table")
            continue
        descriptors.append(table)
        if not entry.get("sources"):
            failures.append(f"{display}: manifest entry has no sources")
        for descriptor in descriptors:
            path = project_root / str(descriptor["path"])
            if not path.is_file():
                failures.append(f"{display}: missing {descriptor['path']}")
                continue
            if sha256(path) != descriptor["sha256"]:
                failures.append(f"{display}: sha256 mismatch for {descriptor['path']}")
            elif path.stat().st_size != descriptor["bytes"]:
                failures.append(f"{display}: byte length mismatch for {descriptor['path']}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    failures = binding_failures(args.manifest, args.project_root)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        print(f"{len(failures)} manifest binding(s) failed", file=sys.stderr)
        return 1
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    bound = sum(len(entry.get("sources", [])) + 1 for entry in manifest.values())
    print(f"OK {bound} manifest bindings verified across {len(manifest)} displays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
