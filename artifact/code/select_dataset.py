"""Select the topic-disjoint, balanced EvidenceGraph-20K claim manifest."""

from __future__ import annotations

import json
from pathlib import Path

from evidencegraph.dataset_builder import select_claims, selection_statistics, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    records = select_claims(PROJECT_ROOT / "data/raw/fever_train.jsonl")
    output = PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl"
    write_jsonl(records, output)
    statistics = selection_statistics(records)
    statistics_path = PROJECT_ROOT / "data/processed/evidencegraph20k_selection_stats.json"
    with statistics_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(statistics, sort_keys=True))


if __name__ == "__main__":
    main()
