from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from graph_memory.retrieval.requests import TextCandidate

MentionSource = Literal["title", "source_ref", "title_prefix", "capitalized_phrase", "alias"]

_CAPITALIZED_TOKEN = r"[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?"
_CAPITALIZED_SPAN_RE = re.compile(rf"\b{_CAPITALIZED_TOKEN}(?:\s+(?:and|of|the|de|van|{_CAPITALIZED_TOKEN}))*")
_TITLE_PREFIX_RE = re.compile(r"^\s*(?P<title>[A-Z][^:]{1,80})\s*:\s+\S")
_PARENTHETICAL_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s*\([^)]*\)\s*$")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "who",
        "which",
    }
)


@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    name: str
    normalized_name: str
    candidate_id: str
    mention_text: str
    source: MentionSource


@dataclass(frozen=True)
class CatalogEntity:
    entity_id: str
    name: str
    normalized_name: str
    entity_type: str
    description: str
    candidate_ids: tuple[str, ...]
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]


@dataclass(frozen=True)
class EntityCatalog:
    entities: tuple[CatalogEntity, ...]


def normalize_entity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def title_aliases(title: str) -> tuple[str, ...]:
    title = title.strip()
    if not title:
        return ()
    aliases = [title]
    match = _PARENTHETICAL_SUFFIX_RE.match(title)
    if match is not None:
        base = match.group("base").strip()
        if base:
            aliases.append(base)
    return _unique_texts(aliases)


def extract_candidate_mentions(candidates: Sequence[TextCandidate]) -> tuple[EntityMention, ...]:
    mentions: list[EntityMention] = []
    seen: set[tuple[str, str, str, MentionSource]] = set()
    for candidate in candidates:
        title = _metadata_text(candidate.metadata, "title")
        title_entity_id = _entity_id(title) if title else None
        for alias in title_aliases(title):
            source: MentionSource = "title" if alias == title else "alias"
            entity_id = title_entity_id or _entity_id(alias)
            _append_mention(mentions, seen, candidate, alias, source, entity_id=entity_id)

        source_ref = _metadata_text(candidate.metadata, "source_ref")
        if source_ref:
            _append_mention(mentions, seen, candidate, source_ref, "source_ref")

        title_prefix = _visible_title_prefix(candidate.text)
        if title_prefix:
            _append_mention(mentions, seen, candidate, title_prefix, "title_prefix")

        title_norms = {normalize_entity_text(alias) for alias in title_aliases(title)}
        for span in _capitalized_spans(candidate.text):
            if normalize_entity_text(span) in title_norms:
                continue
            _append_mention(mentions, seen, candidate, span, "capitalized_phrase")
    return tuple(mentions)


def build_entity_catalog(candidates: Sequence[TextCandidate]) -> EntityCatalog:
    extracted_mentions = extract_candidate_mentions(candidates)
    alias_owner = _unique_alias_owners(extracted_mentions)
    grouped: dict[str, list[EntityMention]] = {}
    for mention in extracted_mentions:
        entity_id = alias_owner.get(mention.normalized_name, mention.entity_id)
        grouped.setdefault(entity_id, []).append(mention)

    entities = []
    for entity_id, mentions in grouped.items():
        preferred = min(mentions, key=_mention_preference)
        aliases = _unique_texts(mention.name for mention in mentions)
        normalized_aliases = _unique_texts(normalize_entity_text(alias) for alias in aliases)
        candidate_ids = tuple(sorted({mention.candidate_id for mention in mentions}))
        entities.append(
            CatalogEntity(
                entity_id=entity_id,
                name=preferred.name,
                normalized_name=preferred.normalized_name,
                entity_type=_entity_type(preferred.source),
                description=preferred.name,
                candidate_ids=candidate_ids,
                aliases=aliases,
                normalized_aliases=normalized_aliases,
            )
        )
    return EntityCatalog(entities=tuple(sorted(entities, key=lambda entity: entity.entity_id)))


def link_query_entities(query_text: str, catalog: EntityCatalog) -> tuple[CatalogEntity, ...]:
    query_norm = normalize_entity_text(query_text)
    exact = [
        entity
        for entity in catalog.entities
        if query_norm == entity.normalized_name or query_norm in entity.normalized_aliases
    ]
    if exact:
        return tuple(sorted(exact, key=lambda entity: entity.entity_id))

    matches: list[tuple[int, int, str, CatalogEntity]] = []
    for entity in catalog.entities:
        for alias in entity.normalized_aliases:
            if not alias:
                continue
            position = _word_substring_position(query_norm, alias)
            if position >= 0:
                matches.append((position, -len(alias), entity.entity_id, entity))
                break
    return tuple(match[3] for match in sorted(matches))


def _append_mention(
    mentions: list[EntityMention],
    seen: set[tuple[str, str, str, MentionSource]],
    candidate: TextCandidate,
    name: str,
    source: MentionSource,
    *,
    entity_id: str | None = None,
) -> None:
    name = name.strip()
    normalized = normalize_entity_text(name)
    if not normalized or normalized in _STOPWORDS:
        return
    entity_id = entity_id or _entity_id(name)
    key = (entity_id, candidate.item_id, normalized, source)
    if key in seen:
        return
    seen.add(key)
    mentions.append(
        EntityMention(
            entity_id=entity_id,
            name=name,
            normalized_name=normalized,
            candidate_id=candidate.item_id,
            mention_text=name,
            source=source,
        )
    )


def _capitalized_spans(text: str) -> tuple[str, ...]:
    spans = []
    for match in _CAPITALIZED_SPAN_RE.finditer(text):
        span = " ".join(match.group(0).split())
        normalized = normalize_entity_text(span)
        if not normalized or normalized in _STOPWORDS:
            continue
        spans.append(span)
    return _unique_texts(spans)


def _entity_id(text: str) -> str:
    normalized = normalize_entity_text(text)
    if normalized:
        return f"e:{normalized.replace(' ', '-')}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"e:{digest}"


def _entity_type(source: MentionSource) -> str:
    if source == "title":
        return "document_title"
    if source == "source_ref":
        return "source_ref"
    if source == "title_prefix":
        return "title_prefix"
    if source == "alias":
        return "alias"
    return "mention"


def _mention_preference(mention: EntityMention) -> tuple[int, str]:
    priority = {
        "title": 0,
        "source_ref": 1,
        "alias": 2,
        "title_prefix": 3,
        "capitalized_phrase": 4,
    }[mention.source]
    return (priority, mention.name)


def _metadata_text(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    return value if isinstance(value, str) else ""


def _unique_alias_owners(mentions: Sequence[EntityMention]) -> dict[str, str]:
    owners_by_alias: dict[str, set[str]] = {}
    for mention in mentions:
        if mention.source not in {"title", "alias"}:
            continue
        owners_by_alias.setdefault(mention.normalized_name, set()).add(mention.entity_id)
    return {
        normalized_alias: next(iter(entity_ids))
        for normalized_alias, entity_ids in owners_by_alias.items()
        if len(entity_ids) == 1
    }


def _unique_texts(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = normalize_entity_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return tuple(result)


def _visible_title_prefix(text: str) -> str:
    match = _TITLE_PREFIX_RE.match(text)
    if match is None:
        return ""
    return match.group("title").strip()


def _word_substring_position(haystack: str, needle: str) -> int:
    match = re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", haystack)
    return -1 if match is None else match.start()


__all__ = [
    "CatalogEntity",
    "EntityCatalog",
    "EntityMention",
    "build_entity_catalog",
    "extract_candidate_mentions",
    "link_query_entities",
    "normalize_entity_text",
    "title_aliases",
]
