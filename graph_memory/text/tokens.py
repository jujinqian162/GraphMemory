from __future__ import annotations

import re

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

