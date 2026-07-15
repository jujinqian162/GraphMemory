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
    EntityKnowledgeGraph,
    EntityKnowledgeGraphEntity,
    EntityKnowledgeGraphRelation,
    GraphRAGRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.text.tokens import content_tokens

_CAPITALIZED_TOKEN = r"[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?"
_CAPITALIZED_SPAN = re.compile(
    rf"\b{_CAPITALIZED_TOKEN}(?:\s+(?:(?:and|of|the|de|van)\s+)?{_CAPITALIZED_TOKEN})*"
)
_ALIAS_SEPARATOR = re.compile(r"\s*[|;,]\s*")


@dataclass(frozen=True)
class _EntityMention:
    name: str
    normalized_name: str
    aliases: tuple[str, ...]
    candidate_id: str
    priority: int


def normalize_entity_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def build_graphrag_request(
    request: TextRankingRequest,
    config: GraphRAGConfig,
) -> GraphRAGRequest:
    graph = build_entity_knowledge_graph(request.candidates, config=config)
    return GraphRAGRequest(
        task_id=request.task_id,
        query_text=request.query_text,
        candidates=request.candidates,
        knowledge_graph=graph,
    )


def build_entity_knowledge_graph(
    candidates: Sequence[TextCandidate],
    *,
    config: GraphRAGConfig,
) -> EntityKnowledgeGraph:
    candidate_list = tuple(candidates)
    mentions = [
        mention
        for candidate in candidate_list
        for mention in _candidate_mentions(candidate, config=config)
    ]
    alias_owner = _unambiguous_alias_owners(mentions)
    grouped: dict[str, list[_EntityMention]] = defaultdict(list)
    for mention in mentions:
        grouped[_canonical_name(mention, alias_owner)].append(mention)

    entities: list[EntityKnowledgeGraphEntity] = []
    entity_id_by_normalized_name: dict[str, str] = {}
    for normalized_name, group in sorted(grouped.items()):
        preferred = min(group, key=lambda item: (item.priority, item.name))
        aliases = {
            alias
            for mention in group
            for alias in (mention.normalized_name, *mention.aliases)
            if alias
        }
        candidate_ids = tuple(sorted({mention.candidate_id for mention in group}))
        entity_id = _entity_id(normalized_name)
        entity_id_by_normalized_name[normalized_name] = entity_id
        entities.append(
            EntityKnowledgeGraphEntity(
                entity_id=entity_id,
                name=preferred.name,
                normalized_name=normalized_name,
                normalized_aliases=tuple(sorted(aliases)),
                description=preferred.name,
                candidate_ids=candidate_ids,
            )
        )

    relation_candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    entity_ids_by_candidate: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        entity_ids_by_candidate[mention.candidate_id].add(
            entity_id_by_normalized_name[_canonical_name(mention, alias_owner)]
        )
    entity_frequency = {
        entity.entity_id: len(entity.candidate_ids) for entity in entities
    }
    for candidate_id, entity_ids in entity_ids_by_candidate.items():
        ordered = sorted(entity_ids)
        for index, source in enumerate(ordered):
            for target in ordered[index + 1 :]:
                relation_candidates[(source, target)].add(candidate_id)

    relations = tuple(
        EntityKnowledgeGraphRelation(
            relation_id=f"relation:{source}:{target}",
            source_entity_id=source,
            target_entity_id=target,
            weight=_cooccurrence_weight(
                len(candidate_ids),
                entity_frequency[source],
                entity_frequency[target],
            ),
            candidate_ids=tuple(sorted(candidate_ids)),
        )
        for (source, target), candidate_ids in sorted(relation_candidates.items())
    )
    return EntityKnowledgeGraph(tuple(entities), relations)


def link_query_entities(
    query_text: str,
    graph: EntityKnowledgeGraph,
) -> tuple[str, ...]:
    normalized_query = normalize_entity_text(query_text)
    matches: list[tuple[int, int, str]] = []
    for entity in graph.entities:
        for alias in entity.normalized_aliases:
            position = _word_substring_position(normalized_query, alias)
            if position >= 0:
                matches.append((position, -len(alias), entity.entity_id))
                break
    return tuple(entity_id for _, _, entity_id in sorted(matches))


def lexical_entity_scores(
    query_text: str,
    graph: EntityKnowledgeGraph,
) -> dict[str, float]:
    query_tokens = set(content_tokens(query_text, keep_short=set()))
    scores: dict[str, float] = {}
    for entity in graph.entities:
        best = 0.0
        for alias in entity.normalized_aliases:
            alias_tokens = set(content_tokens(alias, keep_short=set()))
            if not alias_tokens:
                continue
            best = max(best, len(query_tokens & alias_tokens) / len(alias_tokens))
        if best > 0.0:
            scores[entity.entity_id] = best
    return scores


def _candidate_mentions(
    candidate: TextCandidate,
    *,
    config: GraphRAGConfig,
) -> tuple[_EntityMention, ...]:
    title = _metadata_text(candidate, "title")
    source_ref = _metadata_text(candidate, "source_ref")
    explicit_aliases = tuple(
        normalize_entity_text(alias)
        for alias in _ALIAS_SEPARATOR.split(_metadata_text(candidate, "aliases"))
        if normalize_entity_text(alias)
    )
    mentions: list[_EntityMention] = []
    if title:
        _append_mention(
            mentions,
            candidate.item_id,
            title,
            aliases=tuple(
                alias
                for alias in (normalize_entity_text(source_ref), *explicit_aliases)
                if alias
            ),
            priority=0,
            config=config,
        )
    elif source_ref:
        _append_mention(
            mentions,
            candidate.item_id,
            source_ref,
            aliases=explicit_aliases,
            priority=1,
            config=config,
        )
    for match in _CAPITALIZED_SPAN.finditer(candidate.text):
        _append_mention(
            mentions,
            candidate.item_id,
            match.group(0),
            aliases=(),
            priority=2,
            config=config,
        )
    unique: dict[tuple[str, str], _EntityMention] = {}
    for mention in mentions:
        key = (mention.normalized_name, mention.candidate_id)
        current = unique.get(key)
        if current is None:
            unique[key] = mention
            continue
        preferred = min((current, mention), key=lambda item: (item.priority, item.name))
        unique[key] = _EntityMention(
            name=preferred.name,
            normalized_name=preferred.normalized_name,
            aliases=tuple(sorted(set(current.aliases) | set(mention.aliases))),
            candidate_id=preferred.candidate_id,
            priority=preferred.priority,
        )
    return tuple(sorted(unique.values(), key=lambda item: (item.priority, item.name)))


def _append_mention(
    mentions: list[_EntityMention],
    candidate_id: str,
    name: str,
    *,
    aliases: tuple[str, ...],
    priority: int,
    config: GraphRAGConfig,
) -> None:
    normalized = normalize_entity_text(name)
    if len(normalized) < config.min_entity_length:
        return
    if len(normalized.split()) > config.max_entity_words:
        return
    mentions.append(
        _EntityMention(
            name=" ".join(name.split()),
            normalized_name=normalized,
            aliases=tuple(sorted(set(aliases))),
            candidate_id=candidate_id,
            priority=priority,
        )
    )


def _metadata_text(candidate: TextCandidate, key: str) -> str:
    value = candidate.metadata.get(key)
    return value if isinstance(value, str) else ""


def _unambiguous_alias_owners(
    mentions: Sequence[_EntityMention],
) -> dict[str, str]:
    owners_by_alias: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        if mention.priority > 1:
            continue
        for alias in (mention.normalized_name, *mention.aliases):
            owners_by_alias[alias].add(mention.normalized_name)
    return {
        alias: next(iter(owners))
        for alias, owners in owners_by_alias.items()
        if len(owners) == 1
    }


def _canonical_name(
    mention: _EntityMention,
    alias_owner: dict[str, str],
) -> str:
    return alias_owner.get(mention.normalized_name, mention.normalized_name)


def _entity_id(normalized_name: str) -> str:
    slug = normalized_name.replace(" ", "-")
    if slug:
        return f"entity:{slug}"
    digest = hashlib.sha1(normalized_name.encode("utf-8")).hexdigest()[:12]
    return f"entity:{digest}"


def _cooccurrence_weight(count: int, source_frequency: int, target_frequency: int) -> float:
    return float(count) / math.sqrt(float(source_frequency * target_frequency))


def _word_substring_position(haystack: str, needle: str) -> int:
    if not needle:
        return -1
    match = re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", haystack)
    return -1 if match is None else match.start()


__all__ = [
    "build_entity_knowledge_graph",
    "build_graphrag_request",
    "lexical_entity_scores",
    "link_query_entities",
    "normalize_entity_text",
]
