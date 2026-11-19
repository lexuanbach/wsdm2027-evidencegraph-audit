"""Materialize the EvidenceGraph-20K *validated-paraphrase subset* honestly.

The full 20,000-episode release requires every claim to own the paraphrases its
assigned corruption needs.  Under the frozen bidirectional-NLI plus structural
gate, only a minority of claims passed in the frozen generation budget.  This script
builds episodes for exactly that validated subset with the *unchanged*
``build_episode`` constructor (protocol ``evidencegraph-20k-v4-budget-separated-decoy``,
no Wikidata provenance branch), records the subset denominators explicitly, and
never claims to be the release.  Episodes of claims that lack validated
paraphrases are not fabricated with weaker paraphrases.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from evidencegraph.dataset_builder import (
    build_episode,
    read_jsonl,
    validate_episodes,
    write_jsonl,
)
from evidencegraph.result_registry import source_descriptor
from evidencegraph.wiki_index import WikiIndex
from materialize_dataset import validated_paraphrase_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBSET_PROTOCOL = "evidencegraph-20k-validated-paraphrase-subset-v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_selection.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=PROJECT_ROOT
        / "artifact/runs/paraphrases/production_nfc_v11_qwen15_validated.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_validated_subset.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=PROJECT_ROOT
        / "data/processed/evidencegraph20k_validated_subset_stats.json",
    )
    parser.add_argument(
        "--wiki-index",
        type=Path,
        default=PROJECT_ROOT / "data/processed/fever_wiki.sqlite",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    selection = read_jsonl(args.selection)
    validation_records = read_jsonl(args.validation)
    paraphrases, validator_config = validated_paraphrase_index(
        selection, validation_records
    )
    subset = [record for record in selection if record["claim_id"] in paraphrases]
    excluded = len(selection) - len(subset)
    if not subset:
        raise RuntimeError("no claim has validated paraphrases; nothing to materialize")
    index = WikiIndex(args.wiki_index)
    try:
        episodes = [
            build_episode(record, index, paraphrases[record["claim_id"]])
            for record in subset
        ]
    finally:
        index.close()
    statistics = validate_episodes(episodes, expected_count=len(subset))
    strata = Counter(
        f"{record['split']}:{record['corruption']}:{'true' if record['label'] else 'false'}"
        for record in subset
    )
    statistics.update(
        {
            "subset_protocol": SUBSET_PROTOCOL,
            "is_release": False,
            "selection_rule": (
                "claims whose required paraphrases all passed the frozen "
                "bidirectional-NLI and v5 structural gate; no weaker paraphrase "
                "is substituted and excluded claims are not materialized"
            ),
            "selection_total": len(selection),
            "materialized": len(subset),
            "excluded_without_validated_paraphrases": excluded,
            "split_counts": dict(
                Counter(record["split"] for record in subset)
            ),
            "strata_counts": dict(sorted(strata.items())),
            "validator_model": validator_config[0],
            "validator_revision": validator_config[1],
            "validator_protocol": validator_config[2],
            "validator_threshold": validator_config[3],
            "construction_sources": [
                source_descriptor(args.selection, PROJECT_ROOT),
                source_descriptor(args.validation, PROJECT_ROOT),
            ],
        }
    )
    write_jsonl([episode.to_dict() for episode in episodes], args.output)
    statistics["output"] = source_descriptor(args.output, PROJECT_ROOT)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.stats.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "materialized": len(subset),
                "excluded": excluded,
                "split_counts": statistics["split_counts"],
                "family_counts": statistics.get("family_counts"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
