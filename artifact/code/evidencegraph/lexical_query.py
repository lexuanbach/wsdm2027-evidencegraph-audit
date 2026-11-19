"""Frozen query preprocessing shared by the large SQLite/FTS indices."""

from __future__ import annotations

import re


CONTENT_QUERY_PROTOCOL = "content-terms-stopwords-v1"
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)
QUERY_TOKEN = re.compile(r"[^\W_]+(?:[.'’-][^\W_]+)*", flags=re.UNICODE)


def content_query_tokens(query: str, *, maximum_terms: int = 24) -> list[str]:
    """Return ordered content terms, falling back if a query is all stopwords."""
    if maximum_terms < 1:
        raise ValueError("maximum query terms must be positive")
    all_tokens = [token.lower() for token in QUERY_TOKEN.findall(query)]
    tokens = [token for token in all_tokens if token not in QUERY_STOPWORDS]
    return (tokens or all_tokens)[:maximum_terms]
