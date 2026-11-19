from __future__ import annotations

import json

import pytest

from evidencegraph.averitec_store import (
    AveritecIndex,
    averitec_query_tokens,
    chunk_passages,
    create_averitec_index,
    iter_documents,
)


def _store(tmp_path):
    directory = tmp_path / "output_dev"
    directory.mkdir()
    records = [
        {
            "claim_id": "0",
            "type": "gold",
            "query": "_",
            "url": "https://example.test/a",
            "url2text": ["Alpha evidence.", ""],
        },
        {
            "claim_id": "0",
            "type": "question",
            "query": "beta query",
            "url": "https://example.test/b",
            "url2text": ["Beta evidence."],
        },
    ]
    (directory / "0.json").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return tmp_path


def test_averitec_json_suffix_is_parsed_as_jsonl(tmp_path) -> None:
    store = _store(tmp_path)
    assert len(list(iter_documents(store / "output_dev/0.json"))) == 2


def test_release_fragments_are_coalesced_without_losing_indices() -> None:
    chunks = list(chunk_passages(["alpha", "", "beta", "x" * 110], maximum_characters=100))
    assert chunks == [(0, 2, "alpha beta"), (3, 3, "x" * 110)]


def test_claim_store_query_tokens_drop_only_frozen_stopwords() -> None:
    assert averitec_query_tokens(
        "In a letter to Steve Jobs, Sean Connery refused an Apple commercial."
    ) == ["letter", "steve", "jobs", "sean", "connery", "refused", "apple", "commercial"]
    assert averitec_query_tokens("the and", maximum_terms=1) == ["the"]


def test_build_and_claim_scoped_search(tmp_path) -> None:
    store = _store(tmp_path)
    database = tmp_path / "store.sqlite"
    counts = create_averitec_index(store, database, expected_claims=1)
    assert counts == {
        "claims": 1,
        "documents": 2,
        "passages": 2,
        "raw_passages": 3,
        "empty_passages_skipped": 1,
    }
    index = AveritecIndex(database)
    try:
        assert index.counts() == {"claims": 1, "documents": 2, "passages": 2}
        result = index.search("0", "alpha", limit=1)[0]
        assert result["url"] == "https://example.test/a"
        assert result["text"] == "Alpha evidence."
    finally:
        index.close()


def test_store_rejects_claim_id_mismatch(tmp_path) -> None:
    store = _store(tmp_path)
    path = store / "output_dev/0.json"
    path.write_text(
        json.dumps(
            {
                "claim_id": "1",
                "type": "gold",
                "query": "_",
                "url": "https://example.test",
                "url2text": ["Evidence."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="claim ID"):
        list(iter_documents(path))
