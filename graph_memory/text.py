from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "through",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whom",
    "with",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def content_tokens(text: str, keep_short: set[str] | None = None) -> list[str]:
    keep_short_normalized = {token.lower() for token in (keep_short or set())}
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0)
        if token in STOPWORDS:
            continue
        if len(token) <= 2 and token not in keep_short_normalized:
            continue
        tokens.append(token)
    return tokens


def compute_idf(documents: list[str]) -> dict[str, float]:
    document_count = len(documents)
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(content_tokens(document)))
    return {
        token: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def lexical_score(
    query: str,
    passage: str,
    idf: dict[str, float],
    title_aliases: set[str] | None = None,
    query_entities: set[str] | None = None,
    passage_entities: set[str] | None = None,
) -> float:
    query_tokens = set(content_tokens(query, keep_short=_short_tokens(title_aliases)))
    passage_tokens = set(content_tokens(passage, keep_short=_short_tokens(title_aliases)))
    shared_tokens = query_tokens & passage_tokens
    token_score = sum(idf.get(token, 1.0) for token in shared_tokens)

    query_lower = query.lower()
    passage_lower = passage.lower()
    alias_count = sum(
        1 for alias in (title_aliases or set()) if alias and alias in query_lower and alias in passage_lower
    )
    entity_overlap = set(query_entities or set()) & set(passage_entities or set())
    return token_score + 1.5 * alias_count + 2.0 * len(entity_overlap)


def _short_tokens(aliases: set[str] | None) -> set[str]:
    keep: set[str] = set()
    for alias in aliases or set():
        keep.update(token for token in alias.split() if len(token) <= 2)
    return keep
