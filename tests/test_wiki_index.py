from __future__ import annotations

import json

from evidencegraph.wiki_index import WikiIndex, create_index, parse_lines
from evidencegraph.wiki_title_index import create_title_index


def test_parse_lines_preserves_text_and_link_metadata() -> None:
    rows = list(parse_lines("Page", '0\tFirst sentence.\tlink\tTarget\n1\tSecond sentence.'))
    assert rows[0][:3] == ("Page", 0, "First sentence.")
    assert json.loads(rows[0][3]) == ["link", "Target"]


def test_parse_lines_keeps_first_duplicate_line_number() -> None:
    rows = list(parse_lines("Page", "0\tFirst.\n0\tConflicting duplicate."))
    assert [row[2] for row in rows] == ["First."]


def test_build_lookup_and_search(tmp_path) -> None:
    shard_dir = tmp_path / "wiki" / "wiki-pages"
    shard_dir.mkdir(parents=True)
    page = {"id": "Quokka", "text": "A quokka is a marsupial.", "lines": "0\tA quokka is a marsupial."}
    (shard_dir / "wiki-001.jsonl").write_text(json.dumps(page) + "\n", encoding="utf-8")
    database = tmp_path / "wiki.sqlite"
    counts = create_index(tmp_path / "wiki", database)
    create_title_index(database, tmp_path / "wiki_titles.sqlite")
    assert counts == {
        "pages": 1,
        "sentences": 1,
        "shards": 1,
        "duplicate_line_rows_skipped": 0,
    }
    index = WikiIndex(database)
    try:
        assert index.get_sentence("Quokka", 0) == "A quokka is a marsupial."
        assert index.sentence_count() == 1
        assert index.get_sentence_by_rowid(1) == {
            "page_id": "Quokka",
            "line_number": 0,
            "text": "A quokka is a marsupial.",
        }
        assert index.search("quokka", limit=1)[0]["page_id"] == "Quokka"
        assert index.query_terms("the quokka") == ["quokka"]
    finally:
        index.close()


def test_lookup_normalizes_decomposed_unicode_page_id(tmp_path) -> None:
    shard_dir = tmp_path / "wiki" / "wiki-pages"
    shard_dir.mkdir(parents=True)
    page = {
        "id": "José_María_Chacón",
        "text": "Governor.",
        "lines": "0\tHe was governor.",
    }
    (shard_dir / "wiki-001.jsonl").write_text(
        json.dumps(page, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    database = tmp_path / "wiki.sqlite"
    create_index(tmp_path / "wiki", database)
    index = WikiIndex(database)
    try:
        assert index.get_sentence("Jose\u0301_Mari\u0301a_Chaco\u0301n", 0) == "He was governor."
    finally:
        index.close()


def test_search_expands_links_from_title_matched_pages(tmp_path) -> None:
    shard_dir = tmp_path / "wiki" / "wiki-pages"
    shard_dir.mkdir(parents=True)
    pages = [
        {
            "id": "Alice",
            "text": "Alice links to Bob.",
            "lines": "0\tAlice links to an associate.\tan associate\tBob",
        },
        {
            "id": "Bob",
            "text": "Bob has the hidden evidence.",
            "lines": "0\tBob has the hidden evidence.",
        },
    ]
    (shard_dir / "wiki-001.jsonl").write_text(
        "".join(json.dumps(page) + "\n" for page in pages), encoding="utf-8"
    )
    database = tmp_path / "wiki.sqlite"
    create_index(tmp_path / "wiki", database)
    create_title_index(database, tmp_path / "wiki_titles.sqlite")
    index = WikiIndex(database)
    try:
        results = index.search("Alice has hidden evidence", limit=10)
        assert results[0]["page_id"] == "Alice"
        assert any(
            row["page_id"] == "Bob"
            and row["retrieval_stage"] == "hyperlink-hop-1"
            for row in results
        )
    finally:
        index.close()
