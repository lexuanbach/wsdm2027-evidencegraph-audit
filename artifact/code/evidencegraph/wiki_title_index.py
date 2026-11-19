"""Compact page-title retrieval over the frozen FEVER Wikipedia snapshot."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from .lexical_query import QUERY_STOPWORDS, QUERY_TOKEN


TITLE_QUERY_PROTOCOL = "page-title-entity-phrases-idf-fusion-v2"
TITLE_MAXIMUM_QUERY_TERMS = 8
TITLE_CANDIDATES_PER_TERM = 100
TITLE_MAXIMUM_PHRASES = 40
TITLE_CONNECTORS = frozenset({"da", "de", "del", "of", "the", "van", "von"})


class WikiTitleIndex:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE VIRTUAL TABLE temp.title_query_fts USING "
            "fts5(text, tokenize='porter unicode61')"
        )
        self.connection.execute(
            "CREATE VIRTUAL TABLE temp.title_query_vocab USING "
            "fts5vocab(title_query_fts, 'row')"
        )
        self.connection.execute(
            "CREATE VIRTUAL TABLE temp.title_corpus_vocab USING "
            "fts5vocab(main, titles_fts, 'row')"
        )
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM titles_fts"
        ).fetchone()
        self.title_count = int(row["count"])

    def close(self) -> None:
        self.connection.close()

    def query_terms(self, query: str) -> list[tuple[str, int]]:
        self.connection.execute("DELETE FROM temp.title_query_fts")
        self.connection.execute(
            "INSERT INTO temp.title_query_fts(text) VALUES (?)", (query,)
        )
        tokenized = [
            str(row[0])
            for row in self.connection.execute(
                "SELECT term FROM temp.title_query_vocab ORDER BY term"
            )
        ]
        if not tokenized:
            return []
        placeholders = ",".join("?" for _ in tokenized)
        frequencies = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                "SELECT term, doc FROM temp.title_corpus_vocab "
                f"WHERE term IN ({placeholders})",
                tokenized,
            )
        }
        return sorted(
            frequencies.items(), key=lambda item: (item[1], item[0])
        )[:TITLE_MAXIMUM_QUERY_TERMS]

    def search(self, query: str, limit: int = 30) -> list[dict[str, str | float]]:
        if limit < 1:
            raise ValueError("title retrieval limit must be positive")
        scores: dict[str, float] = {}
        ranks: dict[str, int] = {}
        for phrase in _query_phrases(query):
            rows = self.connection.execute(
                "SELECT page_id FROM titles_fts WHERE titles_fts MATCH ? "
                "ORDER BY bm25(titles_fts) ASC LIMIT 20",
                (f'"{phrase}"',),
            ).fetchall()
            phrase_weight = 20.0 + 5.0 * len(phrase.split())
            for rank, row in enumerate(rows, start=1):
                page_id = str(row["page_id"])
                scores[page_id] = scores.get(page_id, 0.0) + phrase_weight / math.sqrt(rank)
                ranks[page_id] = min(ranks.get(page_id, rank), rank)
        terms = self.query_terms(query)
        for term, frequency in terms:
            idf = math.log((self.title_count + 1.0) / (frequency + 1.0))
            rows = self.connection.execute(
                "SELECT page_id FROM titles_fts WHERE titles_fts MATCH ? "
                "ORDER BY bm25(titles_fts) ASC LIMIT ?",
                (f'"{term}"', TITLE_CANDIDATES_PER_TERM),
            ).fetchall()
            for rank, row in enumerate(rows, start=1):
                page_id = str(row["page_id"])
                scores[page_id] = scores.get(page_id, 0.0) + idf / math.sqrt(rank)
                ranks[page_id] = min(ranks.get(page_id, rank), rank)
        ranked = sorted(
            scores,
            key=lambda page_id: (-scores[page_id], ranks[page_id], len(page_id), page_id),
        )
        return [
            {"page_id": page_id, "score": scores[page_id]}
            for page_id in ranked[:limit]
        ]


def _query_phrases(query: str) -> list[str]:
    matches = list(QUERY_TOKEN.finditer(query))
    tokens = []
    for match in matches:
        token = match.group(0)
        lowered = token.lower()
        if lowered.endswith(("'s", "’s")):
            lowered = lowered[:-2]
        tokens.append(lowered)
    phrases = _entity_phrases(matches)
    for width in range(min(6, len(tokens)), 1, -1):
        for start in range(0, len(tokens) - width + 1):
            span = tokens[start : start + width]
            if all(token in QUERY_STOPWORDS for token in span):
                continue
            phrase = " ".join(span)
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) == TITLE_MAXIMUM_PHRASES:
                return phrases
    return phrases


def _entity_phrases(matches: list[object]) -> list[str]:
    """Extract title-like spans before the generic n-gram budget is exhausted."""
    values = [str(getattr(match, "group")()) for match in matches]

    def notable(value: str) -> bool:
        return bool(value) and value[0].isupper() and value.casefold() not in QUERY_STOPWORDS

    def title_token(value: str) -> bool:
        return bool(value) and value[0].isupper()

    phrases: list[str] = []
    index = 0
    while index < len(values):
        if not notable(values[index]):
            index += 1
            continue
        end = index + 1
        while end < len(values):
            lowered = values[end].casefold()
            if title_token(values[end]):
                end += 1
                continue
            if (
                lowered in TITLE_CONNECTORS
                and end + 1 < len(values)
                and notable(values[end + 1])
            ):
                end += 1
                continue
            break
        phrase = " ".join(value.casefold() for value in values[index:end])
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        index = end
    return phrases


def create_title_index(wiki_database: Path, output: Path) -> dict[str, int | str]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing title index: {output}")
    source = sqlite3.connect(f"file:{wiki_database}?mode=ro", uri=True)
    target = sqlite3.connect(output)
    try:
        target.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE VIRTUAL TABLE titles_fts USING fts5(
                page_id,
                tokenize='porter unicode61'
            );
            """
        )
        cursor = source.execute("SELECT page_id FROM pages ORDER BY rowid")
        count = 0
        while rows := cursor.fetchmany(20_000):
            with target:
                target.executemany(
                    "INSERT INTO titles_fts(page_id) VALUES (?)", rows
                )
            count += len(rows)
        with target:
            target.execute("INSERT INTO titles_fts(titles_fts) VALUES ('optimize')")
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.execute("PRAGMA journal_mode = DELETE")
        expected = int(source.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
        actual = int(target.execute("SELECT COUNT(*) FROM titles_fts").fetchone()[0])
        integrity = str(target.execute("PRAGMA integrity_check").fetchone()[0])
        if count != expected or actual != expected or integrity != "ok":
            raise ValueError(
                f"title index audit failed: inserted={count}, expected={expected}, "
                f"actual={actual}, integrity={integrity}"
            )
        return {
            "protocol": TITLE_QUERY_PROTOCOL,
            "pages": actual,
            "integrity_check": integrity,
            "bytes": output.stat().st_size,
        }
    finally:
        source.close()
        target.close()


def write_title_index_audit(summary: dict[str, int | str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
