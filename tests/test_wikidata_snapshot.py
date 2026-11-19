from __future__ import annotations

import bz2
import json
from pathlib import Path

import pytest

from evidencegraph import wikidata_snapshot as wikidata_module
from evidencegraph.wikidata_snapshot import (
    WIKIDATA_DUMP_BYTES,
    WIKIDATA_DUMP_DATE,
    WIKIDATA_DUMP_MD5,
    WIKIDATA_DUMP_SHA1,
    build_snapshot,
    derive_snapshot_subset,
    fever_page_id_to_title,
    load_provenance_index,
    provenance_usable_page_ids,
    referenced_entity_ids,
)


def _entity(entity_id: str, *, title: str | None = None, claims=None) -> dict:
    record = {
        "id": entity_id,
        "type": "property" if entity_id.startswith("P") else "item",
        "labels": {"en": {"language": "en", "value": f"Label {entity_id}"}},
        "descriptions": {},
        "aliases": {},
        "claims": claims or {},
        "sitelinks": {},
    }
    if title:
        record["sitelinks"]["enwiki"] = {"site": "enwiki", "title": title}
    return record


def _write_dump(path: Path, records: list[dict]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, record in enumerate(records):
            suffix = "," if index + 1 < len(records) else ""
            handle.write(json.dumps(record) + suffix + "\n")
        handle.write("]\n")


def test_fever_page_id_conversion_preserves_parentheses() -> None:
    assert fever_page_id_to_title("Outlander_-LRB-TV_series-RRB-") == "Outlander (TV series)"


def test_fever_page_id_conversion_decodes_colon() -> None:
    assert fever_page_id_to_title("Alien-COLON-_Covenant") == "Alien: Covenant"


def test_referenced_ids_cover_properties_and_item_values() -> None:
    claims = {
        "P31": [
            {
                "mainsnak": {
                    "datavalue": {
                        "value": {"entity-type": "item", "numeric-id": 5}
                    }
                }
            }
        ]
    }
    assert referenced_entity_ids({"claims": claims}) == {"P31", "Q5"}


def test_two_pass_snapshot_extracts_targets_and_reference_terms(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.jsonl"
    manifest.write_text(
        json.dumps({"evidence": [["Outlander_-LRB-TV_series-RRB-", 0]]}) + "\n",
        encoding="utf-8",
    )
    claims = {
        "P31": [
            {
                "mainsnak": {
                    "datavalue": {"value": {"id": "Q5"}}
                }
            }
        ]
    }
    dump = tmp_path / "dump.json.bz2"
    _write_dump(
        dump,
        [
            _entity("P31"),
            _entity("Q5"),
            _entity("Q100", title="Outlander (TV series)", claims=claims),
        ],
    )
    targets = tmp_path / "targets.jsonl"
    terms = tmp_path / "terms.jsonl"
    audit_path = tmp_path / "audit.json"
    audit = build_snapshot(
        dump,
        [manifest],
        targets,
        terms,
        audit_path,
        source={"path": str(dump), "bytes": dump.stat().st_size},
    )

    assert audit["status"] == "complete"
    assert audit["target_titles_matched"] == 1
    term_rows = terms.read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["entity_id"] for line in term_rows} == {
        "P31",
        "Q5",
        "Q100",
    }
    target = json.loads(targets.read_text(encoding="utf-8"))
    assert target["fever_page_ids"] == ["Outlander_-LRB-TV_series-RRB-"]
    assert json.loads(audit_path.read_text(encoding="utf-8"))["status"] == "complete"

    frozen_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    frozen_audit["source"] = {
        "bytes": WIKIDATA_DUMP_BYTES,
        "date": WIKIDATA_DUMP_DATE,
        "license": "CC0-1.0",
        "md5": WIKIDATA_DUMP_MD5,
        "sha1": WIKIDATA_DUMP_SHA1,
    }
    audit_path.write_text(json.dumps(frozen_audit), encoding="utf-8")
    index = load_provenance_index(
        targets,
        terms,
        audit_path,
        required_page_ids={"Outlander_-LRB-TV_series-RRB-"},
    )
    assert index["Outlander_-LRB-TV_series-RRB-"]["entity_id"] == "Q100"
    assert index["Outlander_-LRB-TV_series-RRB-"]["statement_property_ids"] == ["P31"]


def test_snapshot_stops_each_pass_after_complete_requested_set(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "selection.jsonl"
    manifest.write_text(
        json.dumps({"evidence": [["Target", 0]]}) + "\n", encoding="utf-8"
    )
    claims = {
        "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]
    }
    target = _entity("Q100", title="Target", claims=claims)
    passes = 0

    def guarded_entities(_path: Path):
        nonlocal passes
        passes += 1
        records = [target] if passes == 1 else [_entity("P31"), _entity("Q5"), target]
        yield from records
        raise AssertionError("complete requested set was followed past its endpoint")

    monkeypatch.setattr(wikidata_module, "iter_dump_entities", guarded_entities)
    dump = tmp_path / "unused.bz2"
    dump.write_bytes(b"fixture")
    audit = build_snapshot(
        dump,
        [manifest],
        tmp_path / "targets.jsonl",
        tmp_path / "terms.jsonl",
        tmp_path / "audit.json",
        source={"path": str(dump), "bytes": dump.stat().st_size},
    )

    assert passes == 2
    assert audit["status"] == "complete"


def test_final_snapshot_is_derived_from_verified_complete_superset(
    tmp_path: Path,
) -> None:
    superset_manifest = tmp_path / "superset.jsonl"
    final_manifest = tmp_path / "final.jsonl"
    superset_manifest.write_text(
        "\n".join(
            json.dumps({"evidence": [[page, 0]]})
            for page in ("Target_A", "Target_B")
        )
        + "\n",
        encoding="utf-8",
    )
    final_manifest.write_text(
        json.dumps({"evidence": [["Target_A", 0]]}) + "\n",
        encoding="utf-8",
    )
    dump = tmp_path / "dump.json.bz2"
    _write_dump(
        dump,
        [
            _entity("P31"),
            _entity("Q5"),
            _entity(
                "Q100",
                title="Target A",
                claims={
                    "P31": [
                        {"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}
                    ]
                },
            ),
            _entity("Q200", title="Target B"),
        ],
    )
    source_targets = tmp_path / "source_targets.jsonl"
    source_terms = tmp_path / "source_terms.jsonl"
    source_audit = tmp_path / "source_audit.json"
    build_snapshot(
        dump,
        [superset_manifest],
        source_targets,
        source_terms,
        source_audit,
        source={
            "path": str(dump),
            "bytes": WIKIDATA_DUMP_BYTES,
            "date": WIKIDATA_DUMP_DATE,
            "license": "CC0-1.0",
            "md5": WIKIDATA_DUMP_MD5,
            "sha1": WIKIDATA_DUMP_SHA1,
        },
    )
    final_targets = tmp_path / "final_targets.jsonl"
    final_terms = tmp_path / "final_terms.jsonl"
    final_audit = tmp_path / "final_audit.json"

    audit = derive_snapshot_subset(
        final_manifest,
        source_targets,
        source_terms,
        source_audit,
        final_targets,
        final_terms,
        final_audit,
    )

    assert audit["status"] == "complete"
    assert audit["derived_from"]["protocol"] == (
        "checksum-bound-complete-superset-subset-v1"
    )
    targets = [json.loads(line) for line in final_targets.read_text().splitlines()]
    assert [record["entity_id"] for record in targets] == ["Q100"]
    assert targets[0]["fever_page_ids"] == ["Target_A"]
    terms = {json.loads(line)["entity_id"] for line in final_terms.read_text().splitlines()}
    assert terms == {"P31", "Q5", "Q100"}
    index = load_provenance_index(
        final_targets,
        final_terms,
        final_audit,
        required_page_ids={"Target_A"},
    )
    assert index["Target_A"]["entity_id"] == "Q100"


def test_incomplete_parent_can_derive_fully_covered_final_subset(
    tmp_path: Path,
) -> None:
    superset_manifest = tmp_path / "superset.jsonl"
    final_manifest = tmp_path / "final.jsonl"
    superset_manifest.write_text(
        '\n'.join(
            json.dumps({"evidence": [[page, 0]]})
            for page in ("Covered", "Missing")
        ) + "\n",
        encoding="utf-8",
    )
    final_manifest.write_text(
        json.dumps({"evidence": [["Covered", 0]]}) + "\n",
        encoding="utf-8",
    )
    dump = tmp_path / "dump.json.bz2"
    _write_dump(dump, [_entity("Q100", title="Covered")])
    source_targets = tmp_path / "source_targets.jsonl"
    source_terms = tmp_path / "source_terms.jsonl"
    source_audit = tmp_path / "source_audit.json"
    audit = build_snapshot(
        dump,
        [superset_manifest],
        source_targets,
        source_terms,
        source_audit,
        source={
            "path": str(dump),
            "bytes": WIKIDATA_DUMP_BYTES,
            "date": WIKIDATA_DUMP_DATE,
            "license": "CC0-1.0",
            "md5": WIKIDATA_DUMP_MD5,
            "sha1": WIKIDATA_DUMP_SHA1,
        },
    )
    assert audit["status"] == "incomplete"

    final_audit = derive_snapshot_subset(
        final_manifest,
        source_targets,
        source_terms,
        source_audit,
        tmp_path / "final_targets.jsonl",
        tmp_path / "final_terms.jsonl",
        tmp_path / "final_audit.json",
    )

    assert final_audit["status"] == "complete"
    assert final_audit["derived_from"]["parent_status"] == "incomplete"
    assert final_audit["derived_from"]["protocol"] == (
        "checksum-bound-covered-superset-subset-v2"
    )


def test_missing_reference_term_makes_target_page_ineligible(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.jsonl"
    manifest.write_text(
        json.dumps({"evidence": [["Target", 0]]}) + "\n", encoding="utf-8"
    )
    dump = tmp_path / "dump.json.bz2"
    _write_dump(
        dump,
        [
            _entity(
                "Q100",
                title="Target",
                claims={"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]},
            ),
            _entity("Q1000"),
        ],
    )
    targets = tmp_path / "targets.jsonl"
    terms = tmp_path / "terms.jsonl"
    audit_path = tmp_path / "audit.json"
    audit = build_snapshot(
        dump,
        [manifest],
        targets,
        terms,
        audit_path,
        source={
            "path": str(dump),
            "bytes": WIKIDATA_DUMP_BYTES,
            "date": WIKIDATA_DUMP_DATE,
            "license": "CC0-1.0",
            "md5": WIKIDATA_DUMP_MD5,
            "sha1": WIKIDATA_DUMP_SHA1,
        },
    )
    assert audit["status"] == "incomplete"
    assert provenance_usable_page_ids(targets, terms, audit_path) == set()

    with pytest.raises(ValueError, match="target-and-term coverage"):
        derive_snapshot_subset(
            manifest,
            targets,
            terms,
            audit_path,
            tmp_path / "final_targets.jsonl",
            tmp_path / "final_terms.jsonl",
            tmp_path / "final_audit.json",
        )
