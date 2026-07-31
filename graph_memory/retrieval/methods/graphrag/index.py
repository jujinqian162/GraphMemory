from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.retrieval.methods.graphrag.config import GraphRAGConfig
from graph_memory.retrieval.requests import (
    EntityMentionType,
    GraphRAGEntityMention,
    GraphRAGKnowledgeGraph,
    GraphRAGRequest,
    GraphRAGTitleEntityGroup,
    TextCandidate,
    TextRankingRequest,
)

_CAPITALIZED_TOKEN = r"[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?"
_CAPITALIZED_SPAN = re.compile(
    rf"\b{_CAPITALIZED_TOKEN}(?:\s+(?:(?:and|of|the|de|van)\s+)?{_CAPITALIZED_TOKEN})*"
)
_ALIAS_SEPARATOR = re.compile(r"\s*[|;,]\s*")
_STRUCTURED_ENTITY = re.compile(
    r"https?://[^\s\]\[\)\}\>\"']+"
    r"|(?:/[A-Za-z0-9_.@%+,:=~-]+){2,}"
    r"|\b[A-Za-z0-9_.-]{3,}\.(?:jsonl?|md|txt|csv|ya?ml|py|sh|html|pdf)\b"
)
_MIN_ENTITY_LENGTH = 2
_MAX_ENTITY_WORDS = 8
_MAX_BODY_ENTITIES_PER_CANDIDATE = 256
_TITLE_SOURCE_PRIOR = 1.0
_SOURCE_REFERENCE_PRIOR = 0.95
_BODY_MENTION_PRIOR = 0.85
_UNIQUE_ALIAS_CONFIDENCE = 0.9


@dataclass(frozen=True)
class _RawMention:
    candidate_id: str
    surface: str
    normalized_surface: str
    mention_type: EntityMentionType
    source_prior: float
    aliases: tuple[str, ...] = ()


def normalize_entity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def build_graphrag_request(
    request: TextRankingRequest,
    config: GraphRAGConfig,
) -> GraphRAGRequest:
    return GraphRAGRequest(
        task_id=request.task_id,
        query_text=request.query_text,
        candidates=request.candidates,
        knowledge_graph=build_graphrag_knowledge_graph(
            request.candidates, config=config
        ),
    )


def build_graphrag_knowledge_graph(
    candidates: Sequence[TextCandidate],
    *,
    config: GraphRAGConfig,
) -> GraphRAGKnowledgeGraph:
    del config
    candidate_list = tuple(candidates)
    title_mentions = [
        mention
        for candidate in candidate_list
        for mention in _title_mentions(candidate)
    ]
    alias_owners = _alias_owners(title_mentions)
    canonical_title_entities = {
        mention.normalized_surface for mention in title_mentions
    }
    body_mentions = [
        mention for candidate in candidate_list for mention in _body_mentions(candidate)
    ]
    canonical_rows: list[tuple[_RawMention, str, float]] = []
    for mention in title_mentions:
        canonical_rows.append((mention, mention.normalized_surface, 1.0))
    for mention in body_mentions:
        normalized = mention.normalized_surface
        if normalized in canonical_title_entities:
            canonical_rows.append((mention, normalized, 1.0))
            continue
        owners = alias_owners.get(normalized, set())
        if len(owners) == 1:
            canonical_rows.append(
                (mention, next(iter(owners)), _UNIQUE_ALIAS_CONFIDENCE)
            )
        else:
            canonical_rows.append((mention, normalized, 0.0 if owners else 1.0))

    unique_rows: dict[tuple[str, str, str], tuple[_RawMention, str, float]] = {}
    for row in canonical_rows:
        mention, canonical_name, alias_confidence = row
        key = (mention.candidate_id, canonical_name, mention.mention_type)
        current = unique_rows.get(key)
        if current is None or (
            mention.source_prior,
            alias_confidence,
            mention.surface,
        ) > (
            current[0].source_prior,
            current[2],
            current[0].surface,
        ):
            unique_rows[key] = row

    candidates_by_entity: dict[str, set[str]] = defaultdict(set)
    for mention, canonical_name, _alias_confidence in unique_rows.values():
        candidates_by_entity[canonical_name].add(mention.candidate_id)
    candidate_count = len(candidate_list)
    mentions: list[GraphRAGEntityMention] = []
    for key in sorted(unique_rows):
        mention, canonical_name, alias_confidence = unique_rows[key]
        document_frequency = len(candidates_by_entity[canonical_name])
        normalized_idf = _normalized_idf(candidate_count, document_frequency)
        mention_confidence = mention.source_prior * normalized_idf * alias_confidence
        mentions.append(
            GraphRAGEntityMention(
                candidate_id=mention.candidate_id,
                entity_id=_entity_id(canonical_name),
                mention_type=mention.mention_type,
                normalized_surface=canonical_name,
                source_prior=mention.source_prior,
                alias_confidence=alias_confidence,
                entity_document_frequency=document_frequency,
                normalized_idf=normalized_idf,
                mention_confidence=mention_confidence,
            )
        )

    title_candidates: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        if mention.mention_type == "TITLE_ENTITY":
            title_candidates[mention.normalized_surface].add(mention.candidate_id)
    if not title_candidates:
        for mention in mentions:
            title_candidates[mention.normalized_surface].add(mention.candidate_id)
        title_candidates = defaultdict(
            set,
            {
                surface: candidate_ids
                for surface, candidate_ids in title_candidates.items()
                if len(candidate_ids) >= 2
            },
        )
    groups = tuple(
        GraphRAGTitleEntityGroup(
            entity_id=_entity_id(normalized_surface),
            normalized_title_entity=normalized_surface,
            candidate_ids=tuple(sorted(candidate_ids)),
            group_size=len(candidate_ids),
            entity_document_frequency=len(candidates_by_entity[normalized_surface]),
            document_frequency_ratio=(
                len(candidates_by_entity[normalized_surface]) / candidate_count
            ),
        )
        for normalized_surface, candidate_ids in sorted(title_candidates.items())
    )
    return GraphRAGKnowledgeGraph(mentions=tuple(mentions), title_groups=groups)


def link_query_entities(
    query_text: str,
    graph: GraphRAGKnowledgeGraph,
) -> tuple[str, ...]:
    normalized_query = normalize_entity_text(query_text)
    matches: list[tuple[int, int, str]] = []
    surfaces_by_entity: dict[str, set[str]] = defaultdict(set)
    for mention in graph.mentions:
        surfaces_by_entity[mention.entity_id].add(mention.normalized_surface)
    for entity_id, surfaces in surfaces_by_entity.items():
        positions = [
            (position, surface)
            for surface in surfaces
            if (position := _word_substring_position(normalized_query, surface)) >= 0
        ]
        if positions:
            position, surface = min(
                positions, key=lambda item: (item[0], -len(item[1]))
            )
            matches.append((position, -len(surface), entity_id))
    return tuple(entity_id for _position, _length, entity_id in sorted(matches))


def _title_mentions(candidate: TextCandidate) -> tuple[_RawMention, ...]:
    title = _metadata_text(candidate, "title")
    source_ref = _metadata_text(candidate, "source_ref")
    aliases = tuple(
        normalized
        for item in _ALIAS_SEPARATOR.split(_metadata_text(candidate, "aliases"))
        if (normalized := normalize_entity_text(item))
    )
    surface = title or source_ref
    normalized = normalize_entity_text(surface)
    if not _valid_entity(normalized):
        return ()
    source_prior = _TITLE_SOURCE_PRIOR if title else _SOURCE_REFERENCE_PRIOR
    reference_alias = normalize_entity_text(source_ref)
    return (
        _RawMention(
            candidate_id=candidate.item_id,
            surface=" ".join(surface.split()),
            normalized_surface=normalized,
            mention_type="TITLE_ENTITY",
            source_prior=source_prior,
            aliases=tuple(
                sorted(
                    {
                        alias
                        for alias in (reference_alias, *aliases)
                        if alias and alias != normalized
                    }
                )
            ),
        ),
    )


def _body_mentions(candidate: TextCandidate) -> tuple[_RawMention, ...]:
    title = _metadata_text(candidate, "title")
    body = candidate.text
    if title:
        prefix = f"{title}. "
        if body.casefold().startswith(prefix.casefold()):
            body = body[len(prefix) :]
    mentions: dict[str, _RawMention] = {}
    surfaces = [
        " ".join(match.group(0).split())
        for match in _CAPITALIZED_SPAN.finditer(body)
    ]
    surfaces.extend(
        match.group(0).rstrip(".,;:")
        for match in _STRUCTURED_ENTITY.finditer(body)
    )
    for surface in surfaces:
        normalized = normalize_entity_text(surface)
        if not _valid_entity(normalized):
            continue
        mentions.setdefault(
            normalized,
            _RawMention(
                candidate_id=candidate.item_id,
                surface=surface,
                normalized_surface=normalized,
                mention_type="MENTIONS",
                source_prior=_BODY_MENTION_PRIOR,
            ),
        )
    return tuple(
        mentions[key]
        for key in sorted(mentions)[:_MAX_BODY_ENTITIES_PER_CANDIDATE]
    )


def _alias_owners(title_mentions: Sequence[_RawMention]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for mention in title_mentions:
        result[mention.normalized_surface].add(mention.normalized_surface)
        for alias in mention.aliases:
            result[alias].add(mention.normalized_surface)
    return result


def _normalized_idf(candidate_count: int, document_frequency: int) -> float:
    if candidate_count <= 0:
        return 0.0
    return math.log((candidate_count + 1) / (document_frequency + 1)) / math.log(
        candidate_count + 1
    )


def _metadata_text(candidate: TextCandidate, key: str) -> str:
    value = candidate.metadata.get(key)
    return value if isinstance(value, str) else ""


def _valid_entity(normalized: str) -> bool:
    return (
        len(normalized) >= _MIN_ENTITY_LENGTH
        and len(normalized.split()) <= _MAX_ENTITY_WORDS
    )


def _entity_id(normalized_name: str) -> str:
    slug = normalized_name.replace(" ", "-")
    if slug:
        return f"entity:{slug}"
    digest = hashlib.sha1(normalized_name.encode()).hexdigest()[:12]
    return f"entity:{digest}"


def _word_substring_position(haystack: str, needle: str) -> int:
    if not needle:
        return -1
    match = re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", haystack)
    return -1 if match is None else match.start()


__all__ = [
    "build_graphrag_knowledge_graph",
    "build_graphrag_request",
    "link_query_entities",
    "normalize_entity_text",
]
