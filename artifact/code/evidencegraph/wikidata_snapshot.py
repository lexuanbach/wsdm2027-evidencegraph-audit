"""Extract a date-aligned Wikidata subset for EvidenceGraph provenance."""

from __future__ import annotations

import bz2
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .dataset_builder import read_jsonl


ENTITY_ID = re.compile(r"^[PQ][1-9][0-9]*$")
WIKIDATA_DUMP_DATE = "2017-02-20"
WIKIDATA_DUMP_BYTES = 7_166_563_790
WIKIDATA_DUMP_MD5 = "3e5df7fcf8b5b9c893f403a631c70537"
WIKIDATA_DUMP_SHA1 = "ba880b53975224df7dcdf54fc9099e4882aaee74"
WIKIDATA_DUMP_URL = (
    "https://archive.org/download/wikibase-wikidatawiki-20170220/"
    "wikidata-20170220-all.json.bz2"
)


def fever_page_id_to_title(page_id: str) -> str:
    """Convert FEVER's escaped Wikipedia page ID to a normalized title."""

    value = str(page_id)
    for source, target in (
        ("-LRB-", "("),
        ("-RRB-", ")"),
        ("-LSB-", "["),
        ("-RSB-", "]"),
        ("-LCB-", "{"),
        ("-RCB-", "}"),
        ("-COLON-", ":"),
    ):
        value = value.replace(source, target)
    return unicodedata.normalize("NFC", value.replace("_", " ")).strip()


def target_pages(manifests: Iterable[Path]) -> dict[str, list[str]]:
    """Map normalized enwiki titles to every FEVER page ID that requested it."""

    by_title: dict[str, set[str]] = defaultdict(set)
    for manifest in manifests:
        for record in read_jsonl(manifest):
            for page_id, _line_number in record.get("evidence", []):
                page_id = str(page_id)
                by_title[fever_page_id_to_title(page_id)].add(page_id)
    return {title: sorted(values) for title, values in sorted(by_title.items())}


def iter_dump_entities(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the line-oriented JSON array used by Wikidata entity dumps."""

    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = line.strip()
            if not value or value in {"[", "]"}:
                continue
            if value.endswith(","):
                value = value[:-1]
            try:
                entity = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid Wikidata JSON at compressed-stream line {line_number}"
                ) from error
            if not isinstance(entity, dict) or not ENTITY_ID.fullmatch(
                str(entity.get("id", ""))
            ):
                raise ValueError(f"invalid Wikidata entity at line {line_number}")
            yield entity


def _english_value(container: Any) -> str | None:
    if not isinstance(container, dict):
        return None
    value = container.get("en")
    if not isinstance(value, dict) or not str(value.get("value", "")).strip():
        return None
    return str(value["value"])


def _english_aliases(container: Any) -> list[str]:
    if not isinstance(container, dict) or not isinstance(container.get("en"), list):
        return []
    return sorted(
        {
            str(value["value"])
            for value in container["en"]
            if isinstance(value, dict) and str(value.get("value", "")).strip()
        }
    )


def entity_term(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": str(entity["id"]),
        "entity_type": str(entity.get("type", "")),
        "label_en": _english_value(entity.get("labels")),
        "description_en": _english_value(entity.get("descriptions")),
        "aliases_en": _english_aliases(entity.get("aliases")),
    }


def referenced_entity_ids(value: Any) -> set[str]:
    """Collect Q/P identifiers used anywhere in a retained claims object."""

    output: set[str] = set()
    if isinstance(value, dict):
        output.update(
            str(key)
            for key in value
            if str(key).startswith("P") and ENTITY_ID.fullmatch(str(key))
        )
        identifier = str(value.get("id", ""))
        if ENTITY_ID.fullmatch(identifier):
            output.add(identifier)
        entity_type = str(value.get("entity-type", ""))
        numeric_id = value.get("numeric-id")
        if entity_type in {"item", "property"} and isinstance(numeric_id, int):
            prefix = "Q" if entity_type == "item" else "P"
            output.add(f"{prefix}{numeric_id}")
        for key, child in value.items():
            output.update(referenced_entity_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(referenced_entity_ids(child))
    return output


def _target_record(
    entity: dict[str, Any], *, title: str, page_ids: list[str]
) -> dict[str, Any]:
    return {
        **entity_term(entity),
        "enwiki_title": title,
        "fever_page_ids": page_ids,
        "lastrevid": entity.get("lastrevid"),
        "modified": entity.get("modified"),
        "claims": entity.get("claims", {}),
    }


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _digest(path, "sha256"),
    }


def verify_file_descriptor(descriptor: Any, expected_path: Path) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError("Wikidata snapshot file descriptor is missing")
    if Path(str(descriptor.get("path", ""))).resolve() != expected_path.resolve():
        raise ValueError("Wikidata snapshot descriptor path mismatch")
    if descriptor.get("bytes") != expected_path.stat().st_size:
        raise ValueError("Wikidata snapshot descriptor byte mismatch")
    if descriptor.get("sha256") != _digest(expected_path, "sha256"):
        raise ValueError("Wikidata snapshot descriptor checksum mismatch")


def verify_dump(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size != WIKIDATA_DUMP_BYTES:
        raise ValueError(
            f"Wikidata dump byte mismatch: {size} != {WIKIDATA_DUMP_BYTES}"
        )
    md5 = _digest(path, "md5")
    sha1 = _digest(path, "sha1")
    if md5 != WIKIDATA_DUMP_MD5 or sha1 != WIKIDATA_DUMP_SHA1:
        raise ValueError("Wikidata dump checksum mismatch")
    return {
        "path": str(path),
        "bytes": size,
        "md5": md5,
        "sha1": sha1,
        "date": WIKIDATA_DUMP_DATE,
        "url": WIKIDATA_DUMP_URL,
        "license": "CC0-1.0",
    }


def _write_jsonl_atomic(records: Iterable[dict[str, Any]], output: Path) -> None:
    partial = output.with_name(f".{output.name}.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    partial.replace(output)


def _write_json_atomic(record: dict[str, Any], output: Path) -> None:
    partial = output.with_name(f".{output.name}.partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    partial.replace(output)


def build_snapshot(
    dump_path: Path,
    manifests: list[Path],
    target_output: Path,
    term_output: Path,
    audit_output: Path,
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build target entities and referenced English terms in two dump passes."""

    source = source or verify_dump(dump_path)
    pages = target_pages(manifests)
    matched_titles: set[str] = set()
    targets: list[dict[str, Any]] = []
    reference_ids: set[str] = set()
    for entity in iter_dump_entities(dump_path):
        sitelink = entity.get("sitelinks", {}).get("enwiki", {})
        title = unicodedata.normalize("NFC", str(sitelink.get("title", ""))).strip()
        if title not in pages:
            continue
        matched_titles.add(title)
        targets.append(_target_record(entity, title=title, page_ids=pages[title]))
        reference_ids.update(referenced_entity_ids(entity.get("claims", {})))
        if len(matched_titles) == len(pages):
            break

    target_ids = {str(record["entity_id"]) for record in targets}
    wanted_terms = reference_ids | target_ids
    terms = []
    found_terms: set[str] = set()
    for entity in iter_dump_entities(dump_path):
        entity_id = str(entity["id"])
        if entity_id in wanted_terms:
            terms.append(entity_term(entity))
            found_terms.add(entity_id)
            if len(found_terms) == len(wanted_terms):
                break

    targets.sort(key=lambda record: (record["enwiki_title"], record["entity_id"]))
    terms.sort(key=lambda record: record["entity_id"])
    missing_titles = sorted(set(pages) - matched_titles)
    missing_terms = sorted(wanted_terms - found_terms)
    _write_jsonl_atomic(targets, target_output)
    _write_jsonl_atomic(terms, term_output)
    audit = {
        "status": "complete" if not missing_titles and not missing_terms else "incomplete",
        "source": source,
        "inputs": {"manifests": [file_descriptor(path) for path in manifests]},
        "outputs": {
            "targets": file_descriptor(target_output),
            "terms": file_descriptor(term_output),
        },
        "target_titles_requested": len(pages),
        "target_titles_matched": len(matched_titles),
        "target_entities": len(targets),
        "referenced_entity_ids": len(reference_ids),
        "term_entities_requested": len(wanted_terms),
        "term_entities_found": len(found_terms),
        "missing_titles": missing_titles,
        "missing_term_ids": missing_terms,
        "temporal_policy": (
            "2017-02-20 Wikidata predates the June-2017 FEVER Wikipedia snapshot; "
            "later Wikidata statements are excluded"
        ),
    }
    _write_json_atomic(audit, audit_output)
    return audit


def derive_snapshot_subset(
    manifest: Path,
    source_targets: Path,
    source_terms: Path,
    source_audit: Path,
    target_output: Path,
    term_output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    """Derive an exact manifest subset from a verified, covering superset.

    This avoids re-reading the multi-gigabyte compressed dump after reserve
    replacement.  A parent extraction may be incomplete globally, but only if
    its checksum-bound outputs cover every finalized FEVER page and every Q/P
    term referenced by those pages.  The parent is never relabeled complete.
    """

    source_paths = {
        source_targets.resolve(),
        source_terms.resolve(),
        source_audit.resolve(),
    }
    if {
        target_output.resolve(),
        term_output.resolve(),
        audit_output.resolve(),
    } & source_paths:
        raise ValueError("derived Wikidata outputs must not overwrite source artifacts")
    pages = target_pages([manifest])
    required_page_ids = {
        page_id for page_ids in pages.values() for page_id in page_ids
    }
    usable_pages = provenance_usable_page_ids(
        source_targets,
        source_terms,
        source_audit,
    )
    with source_audit.open(encoding="utf-8") as handle:
        parent_audit = json.load(handle)
    uncovered_pages = sorted(required_page_ids - usable_pages)
    if uncovered_pages:
        raise ValueError(
            "Wikidata superset lacks complete target-and-term coverage for "
            f"{len(uncovered_pages)} finalized FEVER pages"
        )

    retained_targets: list[dict[str, Any]] = []
    covered_pages: set[str] = set()
    for record in read_jsonl(source_targets):
        retained = sorted(
            set(map(str, record.get("fever_page_ids", []))) & required_page_ids
        )
        if not retained:
            continue
        retained_targets.append({**record, "fever_page_ids": retained})
        covered_pages.update(retained)
    missing_pages = sorted(required_page_ids - covered_pages)
    if missing_pages:
        raise ValueError(
            f"Wikidata superset lacks {len(missing_pages)} finalized FEVER pages"
        )

    target_ids = {str(record["entity_id"]) for record in retained_targets}
    reference_ids: set[str] = set()
    for record in retained_targets:
        reference_ids.update(referenced_entity_ids(record.get("claims", {})))
    wanted_terms = target_ids | reference_ids
    source_term_index = {
        str(record["entity_id"]): record for record in read_jsonl(source_terms)
    }
    missing_terms = sorted(wanted_terms - set(source_term_index))
    if missing_terms:
        raise ValueError(
            f"Wikidata superset lacks {len(missing_terms)} finalized reference terms"
        )
    retained_terms = [source_term_index[entity_id] for entity_id in sorted(wanted_terms)]
    retained_targets.sort(
        key=lambda record: (str(record["enwiki_title"]), str(record["entity_id"]))
    )
    _write_jsonl_atomic(retained_targets, target_output)
    _write_jsonl_atomic(retained_terms, term_output)
    parent_complete = parent_audit.get("status") == "complete"
    audit = {
        "status": "complete",
        "source": parent_audit["source"],
        "inputs": {"manifests": [file_descriptor(manifest)]},
        "outputs": {
            "targets": file_descriptor(target_output),
            "terms": file_descriptor(term_output),
        },
        "derived_from": {
            "protocol": (
                "checksum-bound-complete-superset-subset-v1"
                if parent_complete
                else "checksum-bound-covered-superset-subset-v2"
            ),
            "parent_status": parent_audit.get("status"),
            "parent_missing_title_count": len(parent_audit.get("missing_titles", [])),
            "parent_missing_term_count": len(parent_audit.get("missing_term_ids", [])),
            "audit": file_descriptor(source_audit),
            "targets": file_descriptor(source_targets),
            "terms": file_descriptor(source_terms),
        },
        "target_titles_requested": len(pages),
        "target_titles_matched": len(
            {str(record["enwiki_title"]) for record in retained_targets}
        ),
        "target_entities": len(retained_targets),
        "referenced_entity_ids": len(reference_ids),
        "term_entities_requested": len(wanted_terms),
        "term_entities_found": len(retained_terms),
        "missing_titles": [],
        "missing_term_ids": [],
        "temporal_policy": parent_audit.get("temporal_policy", ""),
    }
    _write_json_atomic(audit, audit_output)
    return audit


def _verified_snapshot_audit(
    target_path: Path,
    term_path: Path,
    audit_path: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Verify the pinned source metadata and byte-bound extraction outputs."""

    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    source = audit.get("source", {})
    complete = (
        audit.get("status") == "complete"
        and not audit.get("missing_titles")
        and not audit.get("missing_term_ids")
    )
    if (
        audit.get("status") not in {"complete", "incomplete"}
        or (require_complete and not complete)
        or source.get("date") != WIKIDATA_DUMP_DATE
        or source.get("license") != "CC0-1.0"
        or source.get("bytes") != WIKIDATA_DUMP_BYTES
        or source.get("md5") != WIKIDATA_DUMP_MD5
        or source.get("sha1") != WIKIDATA_DUMP_SHA1
    ):
        qualifier = "complete " if require_complete else ""
        raise ValueError(
            f"Wikidata provenance requires a verified {qualifier}pinned snapshot"
        )
    outputs = audit.get("outputs", {})
    verify_file_descriptor(outputs.get("targets"), target_path)
    verify_file_descriptor(outputs.get("terms"), term_path)
    return audit


def provenance_usable_page_ids(
    target_path: Path,
    term_path: Path,
    audit_path: Path,
) -> set[str]:
    """Return pages whose target and every referenced term are present.

    This is the only recovery boundary for an incomplete parent extraction.
    Missing pages or missing terms make the affected FEVER page ineligible;
    they are never imputed from a newer Wikidata snapshot.
    """

    audit = _verified_snapshot_audit(
        target_path, term_path, audit_path, require_complete=False
    )
    term_ids: set[str] = set()
    for record in read_jsonl(term_path):
        entity_id = str(record.get("entity_id", ""))
        if not ENTITY_ID.fullmatch(entity_id) or entity_id in term_ids:
            raise ValueError("invalid or duplicate Wikidata term entity")
        term_ids.add(entity_id)
    if int(audit.get("term_entities_found", -1)) != len(term_ids):
        raise ValueError("Wikidata term row count disagrees with its audit")

    usable: set[str] = set()
    observed_pages: set[str] = set()
    target_count = 0
    matched_titles: set[str] = set()
    for record in read_jsonl(target_path):
        target_count += 1
        entity_id = str(record.get("entity_id", ""))
        title = str(record.get("enwiki_title", ""))
        claims = record.get("claims")
        page_ids = record.get("fever_page_ids")
        if (
            not ENTITY_ID.fullmatch(entity_id)
            or not title
            or not isinstance(claims, dict)
            or not isinstance(page_ids, list)
            or not page_ids
        ):
            raise ValueError("invalid target entity in Wikidata snapshot")
        page_id_set = {str(page_id) for page_id in page_ids}
        if len(page_id_set) != len(page_ids) or observed_pages & page_id_set:
            raise ValueError("duplicate FEVER page mapping in Wikidata snapshot")
        observed_pages.update(page_id_set)
        matched_titles.add(title)
        wanted_terms = {entity_id} | referenced_entity_ids(claims)
        if wanted_terms <= term_ids:
            usable.update(page_id_set)

    if (
        int(audit.get("target_entities", -1)) != target_count
        or int(audit.get("target_titles_matched", -1)) != len(matched_titles)
    ):
        raise ValueError("Wikidata target row counts disagree with its audit")
    if audit.get("status") == "complete" and usable != observed_pages:
        raise ValueError("complete Wikidata audit contains term-incomplete target pages")
    return usable


def load_provenance_index(
    target_path: Path,
    term_path: Path,
    audit_path: Path,
    *,
    required_page_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Load only a complete, byte-bound, date-aligned snapshot."""

    _verified_snapshot_audit(
        target_path, term_path, audit_path, require_complete=True
    )
    usable_pages = provenance_usable_page_ids(target_path, term_path, audit_path)
    unusable_required = sorted(required_page_ids - usable_pages)
    if unusable_required:
        raise ValueError(
            "complete Wikidata provenance lacks term coverage for "
            f"{len(unusable_required)} required FEVER pages"
        )

    index: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(target_path):
        entity_id = str(record.get("entity_id", ""))
        title = str(record.get("enwiki_title", ""))
        claims = record.get("claims")
        page_ids = record.get("fever_page_ids")
        if (
            not ENTITY_ID.fullmatch(entity_id)
            or not title
            or not isinstance(claims, dict)
            or not isinstance(page_ids, list)
            or not page_ids
        ):
            raise ValueError("invalid target entity in Wikidata snapshot")
        summary = {
            "entity_id": entity_id,
            "enwiki_title": title,
            "lastrevid": record.get("lastrevid"),
            "modified": record.get("modified"),
            "statement_property_ids": sorted(
                str(prop) for prop in claims if ENTITY_ID.fullmatch(str(prop))
            ),
            "snapshot_date": WIKIDATA_DUMP_DATE,
            "license": "CC0-1.0",
        }
        for page_id in page_ids:
            page_id = str(page_id)
            if page_id in index:
                raise ValueError(f"duplicate Wikidata mapping for FEVER page {page_id}")
            index[page_id] = summary
    missing = sorted(required_page_ids - set(index))
    if missing:
        raise ValueError(
            f"complete Wikidata provenance lacks {len(missing)} required FEVER pages"
        )
    return index
