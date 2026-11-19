"""Frozen SQLite/FTS5 index for the official FEVER Wikipedia snapshot."""

from __future__ import annotations

import json
import math
import sqlite3
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from .lexical_query import content_query_tokens
from .benchmark_cases import fever_wiki_page_id
from .wiki_title_index import WikiTitleIndex


WIKI_QUERY_PROTOCOL = "title-entity-page-local-bounded-links-v7"
WIKI_MAXIMUM_QUERY_TERMS = 2
WIKI_TITLE_CANDIDATES = 30
WIKI_LINK_PAGES_PER_HOP = 15
WIKI_LINK_HOPS = 2
WIKI_PAGE_SENTENCE_LIMIT = 50


def parse_lines(page_id: str, lines: str) -> Iterator[tuple[str, int, str, str]]:
    """Yield (page, line number, sentence, link metadata) from FEVER lines."""
    seen_line_numbers: set[int] = set()
    for raw_line in lines.splitlines():
        fields = raw_line.split("\t")
        if len(fields) < 2:
            continue
        try:
            line_number = int(fields[0])
        except ValueError:
            continue
        sentence = fields[1].strip()
        # A small number of official records repeat a line number. Preserve the
        # first occurrence in file order so exact evidence lookup is stable.
        if sentence and line_number not in seen_line_numbers:
            seen_line_numbers.add(line_number)
            yield page_id, line_number, sentence, json.dumps(fields[2:], ensure_ascii=False)


class WikiIndex:
    def __init__(
        self, path: Path, read_only: bool = True, title_index_path: Path | None = None
    ):
        if read_only:
            self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._sentence_count: int | None = None
        inferred_title_path = path.with_name(f"{path.stem}_titles{path.suffix}")
        title_path = title_index_path or inferred_title_path
        self.title_index = WikiTitleIndex(title_path) if title_path.is_file() else None
        self.connection.execute(
            "CREATE VIRTUAL TABLE temp.wiki_query_fts USING "
            "fts5(text, tokenize='porter unicode61')"
        )
        self.connection.execute(
            "CREATE VIRTUAL TABLE temp.wiki_query_vocab USING "
            "fts5vocab(wiki_query_fts, 'row')"
        )
        self.connection.execute(
            "CREATE VIRTUAL TABLE temp.wiki_corpus_vocab USING "
            "fts5vocab(main, sentences_fts, 'row')"
        )

    def close(self) -> None:
        if self.title_index is not None:
            self.title_index.close()
        self.connection.close()

    def get_sentence(self, page_id: str, line_number: int) -> str | None:
        row = self.connection.execute(
            "SELECT text FROM sentences WHERE page_id = ? AND line_number = ?",
            (page_id, line_number),
        ).fetchone()
        normalized = unicodedata.normalize("NFC", page_id)
        if row is None and normalized != page_id:
            row = self.connection.execute(
                "SELECT text FROM sentences WHERE page_id = ? AND line_number = ?",
                (normalized, line_number),
            ).fetchone()
        return None if row is None else str(row["text"])

    def sentence_count(self) -> int:
        if self._sentence_count is None:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM sentences").fetchone()
            self._sentence_count = int(row["count"])
        return self._sentence_count

    def get_sentence_by_rowid(self, rowid: int) -> dict[str, str | int] | None:
        row = self.connection.execute(
            "SELECT page_id, line_number, text FROM sentences WHERE rowid = ?", (rowid,)
        ).fetchone()
        return None if row is None else dict(row)

    def query_terms(self, query: str) -> list[str]:
        """Choose the two rarest Porter terms using the frozen FTS vocabulary."""
        self.connection.execute("DELETE FROM temp.wiki_query_fts")
        self.connection.execute(
            "INSERT INTO temp.wiki_query_fts(text) VALUES (?)", (query,)
        )
        tokenized = [
            str(row[0])
            for row in self.connection.execute(
                "SELECT term FROM temp.wiki_query_vocab ORDER BY term"
            )
        ]
        if not tokenized:
            return []
        placeholders = ",".join("?" for _ in tokenized)
        frequencies = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                "SELECT term, doc FROM temp.wiki_corpus_vocab "
                f"WHERE term IN ({placeholders})",
                tokenized,
            )
        }
        ranked = sorted(
            (term for term in tokenized if term in frequencies),
            key=lambda term: (frequencies[term], term),
        )
        if ranked:
            return ranked[:WIKI_MAXIMUM_QUERY_TERMS]
        return content_query_tokens(query, maximum_terms=WIKI_MAXIMUM_QUERY_TERMS)

    def search(self, query: str, limit: int = 100) -> list[dict[str, str | int | float]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if self.title_index is None:
            raise FileNotFoundError("Wikipedia search requires the frozen page-title index")
        query_tokens = set(content_query_tokens(query, maximum_terms=128))
        title_results = self.title_index.search(query, limit=WIKI_TITLE_CANDIDATES)
        all_records: list[dict[str, str | int | float]] = []
        seen_pages: set[str] = set()
        frontier: list[dict[str, str | int | float]] = []
        for rank, result in enumerate(title_results, start=1):
            page_id = str(result["page_id"])
            page_priority = 3.0 / math.sqrt(rank)
            records = self._rank_page_sentences(
                page_id,
                query_tokens,
                page_priority=page_priority,
                retrieval_stage="title",
            )
            if records:
                seen_pages.add(page_id)
                frontier.extend(records)
                all_records.extend(records)
        for hop in range(1, WIKI_LINK_HOPS + 1):
            linked = self._linked_page_candidates(frontier, query_tokens, seen_pages)
            next_frontier = []
            for link_rank, (page_id, _) in enumerate(
                linked[:WIKI_LINK_PAGES_PER_HOP], start=1
            ):
                # Link scores rank the next frontier, but must not recursively
                # inflate page priority and crowd title-matched pages out of
                # the retained lexical prefix.
                page_priority = 1.0 / ((hop + 1) * math.sqrt(link_rank))
                records = self._rank_page_sentences(
                    page_id,
                    query_tokens,
                    page_priority=page_priority,
                    retrieval_stage=f"hyperlink-hop-{hop}",
                )
                if records:
                    seen_pages.add(page_id)
                    next_frontier.extend(records)
                    all_records.extend(records)
            frontier = next_frontier
            if not frontier:
                break
        return _diverse_sentence_order(all_records, limit)

    def _rank_page_sentences(
        self,
        page_id: str,
        query_tokens: set[str],
        *,
        page_priority: float,
        retrieval_stage: str,
    ) -> list[dict[str, str | int | float]]:
        rows = self.connection.execute(
            "SELECT page_id, line_number, text, links_json FROM sentences "
            "WHERE page_id = ? ORDER BY line_number LIMIT ?",
            (page_id, WIKI_PAGE_SENTENCE_LIMIT),
        ).fetchall()
        records = []
        for row in rows:
            sentence_tokens = set(
                content_query_tokens(str(row["text"]), maximum_terms=256)
            )
            overlap = len(query_tokens & sentence_tokens)
            relevance = overlap / max(1.0, math.sqrt(len(sentence_tokens)))
            relevance += page_priority
            if int(row["line_number"]) == 0:
                relevance += 0.1
            records.append(
                {
                    "page_id": str(row["page_id"]),
                    "line_number": int(row["line_number"]),
                    "text": str(row["text"]),
                    "score": -float(relevance),
                    "retrieval_stage": retrieval_stage,
                    "_relevance": float(relevance),
                    "_page_priority": float(page_priority),
                    "_links_json": str(row["links_json"]),
                }
            )
        return sorted(
            records,
            key=lambda record: (
                -float(record["_relevance"]),
                int(record["line_number"]),
            ),
        )

    def _linked_page_candidates(
        self,
        frontier: list[dict[str, str | int | float]],
        query_tokens: set[str],
        seen_pages: set[str],
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for parent in sorted(
            frontier, key=lambda record: -float(record["_relevance"])
        )[:80]:
            links = json.loads(str(parent["_links_json"]))
            for offset in range(0, len(links) - 1, 2):
                anchor, target = str(links[offset]), str(links[offset + 1])
                target = target.split("#", 1)[0].strip()
                if not target:
                    continue
                page_id = fever_wiki_page_id(target)
                if page_id in seen_pages:
                    continue
                bridge_tokens = set(
                    content_query_tokens(f"{anchor} {target}", maximum_terms=64)
                )
                bridge_overlap = len(query_tokens & bridge_tokens)
                priority = (
                    float(parent["_relevance"])
                    + 0.5 * bridge_overlap
                    + 0.1 / (1 + offset // 2)
                )
                scores[page_id] = max(scores.get(page_id, float("-inf")), priority)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _diverse_sentence_order(
    records: list[dict[str, str | int | float]], limit: int
) -> list[dict[str, str | int | float]]:
    by_page: dict[str, list[dict[str, str | int | float]]] = {}
    for record in records:
        by_page.setdefault(str(record["page_id"]), []).append(record)
    for page_records in by_page.values():
        page_records.sort(
            key=lambda record: (
                -float(record["_relevance"]),
                int(record["line_number"]),
            )
        )
    page_order = sorted(
        by_page,
        key=lambda page_id: (
            -max(float(record["_page_priority"]) for record in by_page[page_id]),
            page_id,
        ),
    )
    output = []
    depth = 0
    while len(output) < limit:
        added = False
        for page_id in page_order:
            page_records = by_page[page_id]
            if depth >= len(page_records):
                continue
            record = dict(page_records[depth])
            for internal in ("_relevance", "_page_priority", "_links_json"):
                record.pop(internal, None)
            output.append(record)
            added = True
            if len(output) == limit:
                return output
        if not added:
            break
        depth += 1
    return output


def create_index(wiki_directory: Path, output: Path) -> dict[str, int]:
    files = sorted((wiki_directory / "wiki-pages").glob("wiki-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no official wiki JSONL shards in {wiki_directory}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {output}")
    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE TABLE pages (
                page_id TEXT PRIMARY KEY,
                text TEXT NOT NULL
            );
            CREATE TABLE sentences (
                page_id TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                links_json TEXT NOT NULL,
                PRIMARY KEY (page_id, line_number)
            );
            CREATE VIRTUAL TABLE sentences_fts USING fts5(
                page_id UNINDEXED,
                line_number UNINDEXED,
                text,
                content='sentences',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );
            """
        )
        page_count = sentence_count = duplicate_line_count = 0
        for shard in files:
            pages, sentences = [], []
            with shard.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON at {shard}:{line_number}") from error
                    page_id = str(record.get("id", ""))
                    if not page_id:
                        continue
                    pages.append((page_id, str(record.get("text", ""))))
                    raw_lines = str(record.get("lines", ""))
                    parsed = list(parse_lines(page_id, raw_lines))
                    valid_raw_count = sum(
                        1
                        for raw_line in raw_lines.splitlines()
                        if len(raw_line.split("\t")) >= 2 and raw_line.split("\t")[1].strip()
                    )
                    duplicate_line_count += valid_raw_count - len(parsed)
                    sentences.extend(parsed)
            with connection:
                connection.executemany("INSERT INTO pages(page_id, text) VALUES (?, ?)", pages)
                connection.executemany(
                    "INSERT INTO sentences(page_id, line_number, text, links_json) VALUES (?, ?, ?, ?)",
                    sentences,
                )
            page_count += len(pages)
            sentence_count += len(sentences)
        with connection:
            connection.execute("INSERT INTO sentences_fts(sentences_fts) VALUES ('rebuild')")
            connection.execute("CREATE INDEX sentence_page_index ON sentences(page_id)")
            connection.execute("PRAGMA optimize")
        return {
            "pages": page_count,
            "sentences": sentence_count,
            "shards": len(files),
            "duplicate_line_rows_skipped": duplicate_line_count,
        }
    finally:
        connection.close()
