from __future__ import annotations

import re
from typing import Any

from graph_memory.text import content_tokens

CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b(?:[A-Z][A-Za-z0-9'_-]*)(?:\s+(?:[A-Z][A-Za-z0-9'_-]*))*")
LEADING_DETERMINERS = ("the ", "a ", "an ")


def title_aliases(title: str) -> set[str]:
    normalized_title = _normalize_entity(title)
    aliases = {normalized_title} if normalized_title else set()
    aliases.update(content_tokens(title))
    return aliases


def heuristic_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in CAPITALIZED_PHRASE_PATTERN.finditer(text):
        normalized = _normalize_entity(match.group(0))
        if normalized:
            entities.add(normalized)
    return entities


def extract_entities(text: str, use_spacy: bool = False, nlp: Any | None = None) -> set[str]:
    entities = heuristic_entities(text)
    if use_spacy and nlp is not None:
        document = nlp(text)
        for entity in getattr(document, "ents", []):
            normalized = _normalize_entity(entity.text)
            if normalized:
                entities.add(normalized)
    return entities


def _normalize_entity(value: str) -> str:
    normalized = " ".join(content_tokens(value, keep_short=set())).lower()
    for determiner in LEADING_DETERMINERS:
        if normalized.startswith(determiner):
            normalized = normalized[len(determiner) :]
    return normalized.strip()
