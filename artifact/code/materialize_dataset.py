"""Build EvidenceGraph-20K only from fully validated paraphrase records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegraph.dataset_builder import (
    build_episode,
    read_jsonl,
    validate_episodes,
    write_jsonl,
)
from evidencegraph.final_artifacts import (
    FINAL_COMBINED_VALIDATION,
    FINAL_WIKIDATA_AUDIT,
    FINAL_WIKIDATA_TARGETS,
    FINAL_WIKIDATA_TERMS,
)
from evidencegraph.paraphrase_job import candidate_requirements
from evidencegraph.result_registry import source_descriptor
from evidencegraph.wiki_index import WikiIndex
from evidencegraph.wikidata_snapshot import load_provenance_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validated_paraphrase_index(
    selection: list[dict], validation_records: list[dict]
) -> tuple[dict[str, list[str]], tuple[str, str, str, float]]:
    selected: dict[str, dict] = {}
    for record in selection:
        claim_id = str(record["claim_id"])
        if claim_id in selected:
            raise ValueError(f"duplicate selection claim ID: {claim_id}")
        selected[claim_id] = record

    validation: dict[str, dict] = {}
    validator_configs: set[tuple[str, str, str, float]] = set()
    for record in validation_records:
        claim_id = str(record["claim_id"])
        if claim_id in validation:
            raise ValueError(f"duplicate validation claim ID: {claim_id}")
        if record.get("status") not in {"validated", "semantic_failure"}:
            raise ValueError(f"invalid validation status for {claim_id}")
        validator_model = str(record.get("validator_model", "")).strip()
        validator_revision = str(record.get("validator_revision", "")).strip()
        validator_protocol = str(record.get("validator_protocol", "")).strip()
        threshold = float(record.get("threshold", -1.0))
        if (
            not validator_model
            or not validator_revision
            or not validator_protocol
            or not 0.0 < threshold <= 1.0
        ):
            raise ValueError(f"invalid validator provenance for {claim_id}")
        validator_configs.add(
            (validator_model, validator_revision, validator_protocol, threshold)
        )
        validation[claim_id] = record
    if len(validator_configs) != 1:
        raise ValueError(
            "materialization requires one validator model, revision, protocol, and threshold"
        )

    paraphrases: dict[str, list[str]] = {}
    for claim_id, source in selected.items():
        result = validation.get(claim_id)
        if result is None or result.get("status") != "validated":
            continue
        if str(result.get("source_record_id")) != str(source["source_record_id"]):
            raise ValueError(f"source record mismatch for {claim_id}")
        required = candidate_requirements(source)[1]
        if int(result.get("required_validated", -1)) != required:
            raise ValueError(f"paraphrase requirement mismatch for {claim_id}")
        values = result.get("paraphrases")
        if (
            not isinstance(values, list)
            or len(values) != required
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len({value.casefold().strip() for value in values}) != len(values)
        ):
            raise ValueError(f"invalid validated paraphrases for {claim_id}")
        paraphrases[claim_id] = values
    return paraphrases, next(iter(validator_configs))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_final_selection.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=FINAL_COMBINED_VALIDATION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=PROJECT_ROOT / "data/processed/evidencegraph20k_stats.json",
    )
    parser.add_argument(
        "--wikidata-targets",
        type=Path,
        default=FINAL_WIKIDATA_TARGETS,
    )
    parser.add_argument(
        "--wikidata-terms",
        type=Path,
        default=FINAL_WIKIDATA_TERMS,
    )
    parser.add_argument(
        "--wikidata-audit",
        type=Path,
        default=FINAL_WIKIDATA_AUDIT,
    )
    parser.add_argument("--expected-count", type=int, default=20_000)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    selection = read_jsonl(args.selection)
    if args.limit is not None:
        selection = selection[: args.limit]
    validation_records = read_jsonl(args.validation)
    paraphrases, validator_config = validated_paraphrase_index(
        selection, validation_records
    )
    missing = [record["claim_id"] for record in selection if record["claim_id"] not in paraphrases]
    if missing:
        raise RuntimeError(
            f"refusing partial materialization: {len(missing)} claims lack sufficient "
            "validated paraphrases for their assigned corruption"
        )
    required_pages = {
        str(page_id)
        for record in selection
        for page_id, _line_number in record["evidence"]
    }
    wikidata = load_provenance_index(
        args.wikidata_targets,
        args.wikidata_terms,
        args.wikidata_audit,
        required_page_ids=required_pages,
    )
    index = WikiIndex(PROJECT_ROOT / "data/processed/fever_wiki.sqlite")
    try:
        episodes = [
            build_episode(
                record,
                index,
                paraphrases[record["claim_id"]],
                wikidata_entities=sorted(
                    {
                        wikidata[str(page_id)]["entity_id"]: wikidata[str(page_id)]
                        for page_id, _line_number in record["evidence"]
                    }.values(),
                    key=lambda entity: str(entity["entity_id"]),
                ),
            )
            for record in selection
        ]
    finally:
        index.close()
    statistics = validate_episodes(episodes, expected_count=args.expected_count)
    statistics["validator_model"] = validator_config[0]
    statistics["validator_revision"] = validator_config[1]
    statistics["validator_protocol"] = validator_config[2]
    statistics["validator_threshold"] = validator_config[3]
    statistics["construction_sources"] = [
        source_descriptor(args.selection, PROJECT_ROOT),
        source_descriptor(args.validation, PROJECT_ROOT),
        source_descriptor(args.wikidata_targets, PROJECT_ROOT),
        source_descriptor(args.wikidata_terms, PROJECT_ROOT),
        source_descriptor(args.wikidata_audit, PROJECT_ROOT),
    ]
    write_jsonl(
        [episode.to_dict() for episode in episodes],
        args.output,
    )
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with args.stats.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(statistics, sort_keys=True))


if __name__ == "__main__":
    main()
