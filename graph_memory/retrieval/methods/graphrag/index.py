from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from graph_memory.retrieval.methods.graphrag.config import GraphRAGConfig
from graph_memory.retrieval.methods.graphrag.noun_phrases import (
    extract_noun_phrases,
    normalize_entity_text,
)
from graph_memory.retrieval.requests import (
    GraphRAGEntity,
    GraphRAGKnowledgeGraph,
    GraphRAGRelation,
    GraphRAGRequest,
    GraphRAGTextUnit,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.text.tokens import content_tokens

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class _RawTextUnit:
    unit_id: str
    text: str
    candidate_ids: tuple[str, ...]


def build_graphrag_request(
    request: TextRankingRequest,
    config: GraphRAGConfig,
    *,
    knowledge_graph: GraphRAGKnowledgeGraph | None = None,
) -> GraphRAGRequest:
    return GraphRAGRequest(
        task_id=request.task_id,
        query_text=request.query_text,
        candidates=request.candidates,
        knowledge_graph=(
            knowledge_graph
            if knowledge_graph is not None
            else build_graphrag_knowledge_graph(request.candidates, config=config)
        ),
    )


def build_graphrag_knowledge_graph(
    candidates: Sequence[TextCandidate],
    *,
    config: GraphRAGConfig,
) -> GraphRAGKnowledgeGraph:
    raw_units = _build_text_units(tuple(candidates), config=config)
    phrases_by_unit = {
        unit.unit_id: extract_noun_phrases(unit.text, config) for unit in raw_units
    }
    units_by_id = {unit.unit_id: unit for unit in raw_units}

    unit_ids_by_phrase: dict[str, set[str]] = defaultdict(set)
    for unit_id, phrases in phrases_by_unit.items():
        for phrase in phrases:
            unit_ids_by_phrase[phrase].add(unit_id)
    surviving_phrases = {
        phrase
        for phrase, unit_ids in unit_ids_by_phrase.items()
        if len(unit_ids) >= config.min_node_frequency
    }

    relation_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    for unit_id, phrases in phrases_by_unit.items():
        kept = sorted(set(phrases) & surviving_phrases)
        for source, target in combinations(kept, 2):
            relation_units[(source, target)].add(unit_id)

    degree: CounterLike = defaultdict(int)
    for source, target in relation_units:
        degree[source] += 1
        degree[target] += 1
    removed = {
        phrase
        for phrase in surviving_phrases
        if degree.get(phrase, 0) < config.min_node_degree
    }
    if config.remove_ego_node and degree:
        removed.add(max(degree, key=lambda phrase: (degree[phrase], phrase)))
    surviving_phrases -= removed
    relation_units = {
        pair: unit_ids
        for pair, unit_ids in relation_units.items()
        if pair[0] in surviving_phrases and pair[1] in surviving_phrases
    }

    raw_weights = _relation_weights(
        relation_units,
        unit_ids_by_phrase=unit_ids_by_phrase,
        normalize=config.normalize_edge_weights,
    )
    positive_weights = {
        pair: weight for pair, weight in raw_weights.items() if weight > 0.0
    }
    if positive_weights and config.min_edge_weight_percentile > 0.0:
        threshold = _percentile(
            tuple(positive_weights.values()), config.min_edge_weight_percentile
        )
        positive_weights = {
            pair: weight
            for pair, weight in positive_weights.items()
            if weight >= threshold
        }

    related_phrases = {phrase for pair in positive_weights for phrase in pair}
    surviving_phrases &= related_phrases
    active_unit_ids = {
        unit_id
        for phrase in surviving_phrases
        for unit_id in unit_ids_by_phrase[phrase]
    }
    text_units = tuple(
        GraphRAGTextUnit(unit_id=unit.unit_id, candidate_ids=unit.candidate_ids)
        for unit in raw_units
        if unit.unit_id in active_unit_ids
    )

    entities = tuple(
        GraphRAGEntity(
            entity_id=_entity_id(phrase),
            name=phrase,
            frequency=len(unit_ids_by_phrase[phrase]),
            text_unit_ids=tuple(sorted(unit_ids_by_phrase[phrase])),
            candidate_ids=tuple(
                sorted(
                    {
                        candidate_id
                        for unit_id in unit_ids_by_phrase[phrase]
                        for candidate_id in units_by_id[unit_id].candidate_ids
                    }
                )
            ),
        )
        for phrase in sorted(surviving_phrases)
    )
    entity_id_by_phrase = {entity.name: entity.entity_id for entity in entities}
    relations = tuple(
        GraphRAGRelation(
            relation_id=_relation_id(source, target),
            source_entity_id=entity_id_by_phrase[source],
            target_entity_id=entity_id_by_phrase[target],
            weight=weight,
            text_unit_ids=tuple(sorted(relation_units[(source, target)])),
            candidate_ids=tuple(
                sorted(
                    {
                        candidate_id
                        for unit_id in relation_units[(source, target)]
                        for candidate_id in units_by_id[unit_id].candidate_ids
                    }
                )
            ),
        )
        for (source, target), weight in sorted(positive_weights.items())
        if source in surviving_phrases and target in surviving_phrases
    )
    return GraphRAGKnowledgeGraph(
        text_units=text_units,
        entities=entities,
        relations=relations,
    )


def link_query_entities(
    query_text: str,
    graph: GraphRAGKnowledgeGraph,
) -> tuple[str, ...]:
    normalized_query = normalize_entity_text(query_text)
    matches: list[tuple[int, int, str]] = []
    for entity in graph.entities:
        position = _word_substring_position(normalized_query, entity.name)
        if position >= 0:
            matches.append((position, -len(entity.name), entity.entity_id))
    return tuple(entity_id for _position, _length, entity_id in sorted(matches))


def lexical_entity_scores(
    query_text: str,
    graph: GraphRAGKnowledgeGraph,
) -> dict[str, float]:
    query_tokens = set(content_tokens(query_text, keep_short=set()))
    scores: dict[str, float] = {}
    for entity in graph.entities:
        entity_tokens = set(content_tokens(entity.name, keep_short=set()))
        if not entity_tokens:
            continue
        score = len(query_tokens & entity_tokens) / len(entity_tokens)
        if score > 0.0:
            scores[entity.entity_id] = score
    return scores


def _build_text_units(
    candidates: tuple[TextCandidate, ...],
    *,
    config: GraphRAGConfig,
) -> tuple[_RawTextUnit, ...]:
    accumulated: dict[str, tuple[str, set[str]]] = {}
    step = config.text_unit_size - config.text_unit_overlap
    for candidate in candidates:
        matches = tuple(_TOKEN.finditer(candidate.text))
        if not matches:
            continue
        for start in range(0, len(matches), step):
            end = min(start + config.text_unit_size, len(matches))
            text = candidate.text[matches[start].start() : matches[end - 1].end()]
            normalized = " ".join(text.split())
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
            unit_id = f"graphrag-unit:{digest}"
            current = accumulated.get(unit_id)
            if current is None:
                accumulated[unit_id] = (text, {candidate.item_id})
            elif current[0] == text:
                current[1].add(candidate.item_id)
            else:
                collision_digest = hashlib.sha256(
                    f"{candidate.item_id}\0{start}\0{text}".encode("utf-8")
                ).hexdigest()[:20]
                accumulated[f"graphrag-unit:{collision_digest}"] = (
                    text,
                    {candidate.item_id},
                )
            if end == len(matches):
                break
    return tuple(
        _RawTextUnit(
            unit_id=unit_id,
            text=text,
            candidate_ids=tuple(sorted(candidate_ids)),
        )
        for unit_id, (text, candidate_ids) in sorted(accumulated.items())
    )


def _relation_weights(
    relation_units: dict[tuple[str, str], set[str]],
    *,
    unit_ids_by_phrase: dict[str, set[str]],
    normalize: bool,
) -> dict[tuple[str, str], float]:
    if not relation_units:
        return {}
    if not normalize:
        return {pair: float(len(unit_ids)) for pair, unit_ids in relation_units.items()}
    total_edge_occurrences = sum(len(unit_ids) for unit_ids in relation_units.values())
    total_node_occurrences = sum(
        len(unit_ids) for unit_ids in unit_ids_by_phrase.values()
    )
    result: dict[tuple[str, str], float] = {}
    for (source, target), unit_ids in relation_units.items():
        joint = len(unit_ids) / total_edge_occurrences
        source_probability = len(unit_ids_by_phrase[source]) / total_node_occurrences
        target_probability = len(unit_ids_by_phrase[target]) / total_node_occurrences
        # FastGraphRAG uses a frequency-scaled PMI. PPR requires non-negative
        # transition weights, so the retrieval adaptation uses its positive part.
        result[(source, target)] = max(
            0.0,
            joint * math.log2(joint / (source_probability * target_probability)),
        )
    return result


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _entity_id(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return f"entity:{digest}"


def _relation_id(source: str, target: str) -> str:
    digest = hashlib.sha1(f"{source}\0{target}".encode("utf-8")).hexdigest()[:16]
    return f"entity-relation:{digest}"


def _word_substring_position(haystack: str, needle: str) -> int:
    if not needle:
        return -1
    match = re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", haystack)
    return -1 if match is None else match.start()


CounterLike = dict[str, int]

__all__ = [
    "build_graphrag_knowledge_graph",
    "build_graphrag_request",
    "lexical_entity_scores",
    "link_query_entities",
]
