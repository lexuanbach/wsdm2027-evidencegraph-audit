from __future__ import annotations

import sqlite3

from evidencegraph.wiki_title_index import WikiTitleIndex, _query_phrases, create_title_index


def test_title_index_retrieves_entity_page_from_claim(tmp_path) -> None:
    wiki = tmp_path / "wiki.sqlite"
    connection = sqlite3.connect(wiki)
    connection.execute("CREATE TABLE pages (page_id TEXT PRIMARY KEY, text TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO pages(page_id, text) VALUES (?, '')",
        [("Yale_University",), ("Supreme_Court_of_the_United_States",), ("Other",)],
    )
    connection.commit()
    connection.close()
    titles = tmp_path / "titles.sqlite"
    summary = create_title_index(wiki, titles)
    assert summary["pages"] == 3
    index = WikiTitleIndex(titles)
    try:
        results = index.search("Yale University's alumni includes 19 justices")
        assert results[0]["page_id"] == "Yale_University"
    finally:
        index.close()


def test_title_phrases_prioritize_entities_and_dotted_initialisms() -> None:
    phrases = _query_phrases(
        "The Fergie song produced by Polow da Don was followed by Life Goes On and M.I.L.F.$."
    )

    assert phrases[:4] == ["fergie", "polow da don", "life goes on", "m.i.l.f"]


def test_title_index_recovers_entity_qualified_and_dotted_titles(tmp_path) -> None:
    wiki = tmp_path / "wiki.sqlite"
    connection = sqlite3.connect(wiki)
    connection.execute("CREATE TABLE pages (page_id TEXT PRIMARY KEY, text TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO pages(page_id, text) VALUES (?, '')",
        [
            ("M.I.L.F._$",),
            ("Life_Goes_On_-LRB-Fergie_song-RRB-",),
            ("Life_Goes_On_-LRB-other_song-RRB-",),
            ("Polow_da_Don",),
            ("Fergie",),
        ],
    )
    connection.commit()
    connection.close()
    titles = tmp_path / "titles.sqlite"
    create_title_index(wiki, titles)
    index = WikiTitleIndex(titles)
    try:
        results = index.search(
            "The Fergie song produced by Polow da Don was followed by Life Goes On and M.I.L.F.$.",
            limit=5,
        )
        page_ids = [row["page_id"] for row in results]
        assert "M.I.L.F._$" in page_ids
        assert "Life_Goes_On_-LRB-Fergie_song-RRB-" in page_ids
    finally:
        index.close()
