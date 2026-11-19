"""Validated SQLite/FTS5 index for the official AVeriTeC knowledge store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .lexical_query import CONTENT_QUERY_PROTOCOL, content_query_tokens

AVERITEC_QUERY_PROTOCOL = f"claim-rowid-bounded-bm25-{CONTENT_QUERY_PROTOCOL}"


def averitec_query_tokens(query: str, *, maximum_terms: int = 24) -> list[str]:
    """Return frozen content-bearing terms for tractable claim-scoped BM25."""
    return content_query_tokens(query, maximum_terms=maximum_terms)


def store_files(root: Path, *, expected_claims: int | None = 500) -> list[Path]:
    directory = root / "output_dev" if (root / "output_dev").is_dir() else root
    files = list(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no AVeriTeC claim stores under {directory}")
    try:
        files.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError("AVeriTeC store filenames must be numeric claim IDs") from error
    claim_ids = [int(path.stem) for path in files]
    if expected_claims is not None and claim_ids != list(range(expected_claims)):
        raise ValueError(
            f"AVeriTeC store must contain claim files 0..{expected_claims - 1}; "
            f"found {len(files)} files"
        )
    return files


def iter_documents(path: Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSONL documents despite the release's `.json` suffix."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            required = {"claim_id", "type", "query", "url", "url2text"}
            if not required.issubset(record):
                raise ValueError(f"{path}:{line_number} lacks required store fields")
            if str(record["claim_id"]) != path.stem:
                raise ValueError(f"{path}:{line_number} claim ID does not match filename")
            if not isinstance(record["url2text"], list) or any(
                not isinstance(value, str) for value in record["url2text"]
            ):
                raise ValueError(f"{path}:{line_number} url2text must be a string list")
            if not isinstance(record["url"], str) or not record["url"].strip():
                raise ValueError(f"{path}:{line_number} has no source URL")
            yield record


def chunk_passages(
    values: list[str], *, maximum_characters: int = 800
) -> Iterator[tuple[int, int, str]]:
    """Coalesce release fragments into deterministic retrieval-sized chunks."""
    if maximum_characters < 100:
        raise ValueError("AVeriTeC chunk size must be at least 100 characters")
    buffered: list[str] = []
    start_index: int | None = None
    end_index: int | None = None
    characters = 0
    for passage_index, raw_text in enumerate(values):
        text = raw_text.strip()
        if not text:
            continue
        added = len(text) + (1 if buffered else 0)
        if buffered and characters + added > maximum_characters:
            assert start_index is not None and end_index is not None
            yield start_index, end_index, " ".join(buffered)
            buffered = []
            start_index = None
            characters = 0
        if start_index is None:
            start_index = passage_index
        buffered.append(text)
        end_index = passage_index
        characters += len(text) + (1 if len(buffered) > 1 else 0)
    if buffered:
        assert start_index is not None and end_index is not None
        yield start_index, end_index, " ".join(buffered)


class AveritecIndex:
    def __init__(self, path: Path, read_only: bool = True):
        if read_only:
            self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._claim_rowid_bounds: dict[str, tuple[int, int] | None] = {}

    def close(self) -> None:
        self.connection.close()

    def search(self, claim_id: str, query: str, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        claim_id = str(claim_id)
        tokens = averitec_query_tokens(query)
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token}"' for token in tokens)
        if claim_id not in self._claim_rowid_bounds:
            bounds = self.connection.execute(
                "SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM passages "
                "WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
            if bounds is None or bounds[0] is None or bounds[1] is None:
                self._claim_rowid_bounds[claim_id] = None
            else:
                lower, upper, count = map(int, bounds)
                if upper - lower + 1 != count:
                    raise RuntimeError(
                        "AVeriTeC claim passages are not rowid-contiguous; "
                        "bounded FTS is unsafe"
                    )
                self._claim_rowid_bounds[claim_id] = (lower, upper)
        rowid_bounds = self._claim_rowid_bounds[claim_id]
        if rowid_bounds is None:
            return []
        rows = self.connection.execute(
            """
            SELECT p.claim_id, d.url, d.document_index,
                   p.chunk_index AS passage_index, p.text,
                   d.source_type, d.source_query,
                   bm25(passages_fts) AS score
            FROM passages_fts
            JOIN passages AS p ON p.rowid = passages_fts.rowid
            JOIN documents AS d ON d.document_id = p.document_id
            WHERE passages_fts MATCH ?
              AND passages_fts.rowid BETWEEN ? AND ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query, rowid_bounds[0], rowid_bounds[1], limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        return {
            "claims": int(
                self.connection.execute(
                    "SELECT COUNT(DISTINCT claim_id) FROM passages"
                ).fetchone()[0]
            ),
            "documents": int(
                self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            ),
            "passages": int(
                self.connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            ),
        }


def create_averitec_index(
    store_root: Path,
    output: Path,
    *,
    expected_claims: int | None = 500,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    files = store_files(store_root, expected_claims=expected_claims)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    documents = passages = raw_passages = empty_passages = 0
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE TABLE documents (
                document_id INTEGER PRIMARY KEY,
                claim_id TEXT NOT NULL,
                document_index INTEGER NOT NULL,
                url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_query TEXT NOT NULL,
                UNIQUE (claim_id, document_index)
            );
            CREATE TABLE passages (
                claim_id TEXT NOT NULL,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                first_passage_index INTEGER NOT NULL,
                last_passage_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (document_id, chunk_index),
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            );
            CREATE INDEX passages_claim_id ON passages(claim_id);
            CREATE VIRTUAL TABLE passages_fts USING fts5(
                claim_id UNINDEXED,
                text,
                content='passages',
                content_rowid='rowid'
            );
            """
        )
        for path in files:
            for document_index, record in enumerate(iter_documents(path)):
                claim_id = str(record["claim_id"])
                url = str(record["url"]).strip()
                source_type = str(record["type"])
                source_query = str(record["query"])
                connection.execute(
                    "INSERT INTO documents(claim_id, document_index, url, source_type, "
                    "source_query) VALUES (?, ?, ?, ?, ?)",
                    (claim_id, document_index, url, source_type, source_query),
                )
                document_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
                documents += 1
                raw_passages += len(record["url2text"])
                empty_passages += sum(not value.strip() for value in record["url2text"])
                for chunk_index, (first_index, last_index, text) in enumerate(
                    chunk_passages(record["url2text"])
                ):
                    connection.execute(
                        "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            claim_id,
                            document_id,
                            chunk_index,
                            first_index,
                            last_index,
                            text,
                        ),
                    )
                    passages += 1
                if documents % 1000 == 0:
                    connection.commit()
                if progress is not None and documents % 10_000 == 0:
                    progress(
                        {
                            "stage": "documents",
                            "documents": documents,
                            "passages": passages,
                            "claim_id": claim_id,
                        }
                    )
        connection.commit()
        if progress is not None:
            progress({"stage": "fts_rebuild", "documents": documents, "passages": passages})
        connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"AVeriTeC index integrity check failed: {integrity}")
        if progress is not None:
            progress({"stage": "complete", "documents": documents, "passages": passages})
    finally:
        connection.close()
    return {
        "claims": len(files),
        "documents": documents,
        "passages": passages,
        "raw_passages": raw_passages,
        "empty_passages_skipped": empty_passages,
    }
