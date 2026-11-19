"""Write normalized official benchmark case manifests with strict counts."""

from __future__ import annotations

import json
from pathlib import Path

from evidencegraph.benchmark_cases import (
    averitec_cases,
    expertqa_cases,
    fever_cases,
    hover_cases,
)
from evidencegraph.dataset_builder import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    external = PROJECT_ROOT / ".external"
    output = PROJECT_ROOT / "data/benchmarks"
    manifests = {
        "fever_dev": fever_cases(PROJECT_ROOT / "data/raw/fever_dev.jsonl"),
        "hover_dev": hover_cases(
            external / "hover-official/data/hover/hover_dev_release_v1.1.json"
        ),
        "averitec_dev": averitec_cases(external / "averitec-official/data/dev.json"),
        "expertqa_domain_test": expertqa_cases(
            external
            / "expertqa-official/data/r2_compiled_all_evidences_autoais_anon.jsonl",
            external / "expertqa-official/data/lfqa/domain_lfqa_test.json",
        ),
    }
    expected = {
        "fever_dev": 19_998,
        "hover_dev": 4_000,
        "averitec_dev": 500,
        "expertqa_domain_test": 243,
    }
    counts = {name: len(cases) for name, cases in manifests.items()}
    if counts != expected:
        raise ValueError(f"benchmark case count mismatch: {counts} != {expected}")
    for name, cases in manifests.items():
        write_jsonl([case.to_dict() for case in cases], output / f"{name}.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "case_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
