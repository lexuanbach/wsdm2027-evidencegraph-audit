from __future__ import annotations

import json

import pytest

from evidencegraph.preflight import preflight_evidence_references
from evidencegraph.wiki_index import WikiIndex, create_index


def _index(tmp_path) -> WikiIndex:
    shard_dir = tmp_path / "wiki" / "wiki-pages"
    shard_dir.mkdir(parents=True)
    page = {"id": "José", "text": "Evidence.", "lines": "0\tEvidence."}
    (shard_dir / "wiki-001.jsonl").write_text(
        json.dumps(page, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    database = tmp_path / "wiki.sqlite"
    create_index(tmp_path / "wiki", database)
    return WikiIndex(database)


def test_preflight_resolves_all_disjoint_cohorts_with_unicode_normalization(tmp_path) -> None:
    index = _index(tmp_path)
    try:
        summary = preflight_evidence_references(
            {
                "primary": [{"claim_id": "p", "evidence": [["Jose\u0301", 0]]}],
                "reserve": [{"claim_id": "r", "evidence": [["José", 0]]}],
            },
            index,
        )
    finally:
        index.close()

    assert summary["total_records"] == 2
    assert summary["total_references"] == 2
    assert summary["missing_references"] == 0


def test_preflight_fails_on_missing_reference(tmp_path) -> None:
    index = _index(tmp_path)
    try:
        with pytest.raises(ValueError, match="1 evidence references do not resolve"):
            preflight_evidence_references(
                {"primary": [{"claim_id": "p", "evidence": [["Missing", 7]]}]},
                index,
            )
    finally:
        index.close()
