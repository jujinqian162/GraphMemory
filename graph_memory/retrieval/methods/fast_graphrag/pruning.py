from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGPruningConfig
from graph_memory.retrieval.requests import FastGraphRAGEntity, FastGraphRAGKnowledgeGraph, FastGraphRAGRelation


def prune_knowledge_graph(
    kg: FastGraphRAGKnowledgeGraph,
    config: FastGraphRAGPruningConfig,
) -> FastGraphRAGKnowledgeGraph:
    kept_entity_ids = _entity_ids_after_node_filters(kg, config)
    relations = tuple(
        relation
        for relation in kg.relations
        if relation.source_entity_id in kept_entity_ids and relation.target_entity_id in kept_entity_ids
    )
    relations = _filter_relations_by_weight_percentile(relations, config.min_edge_weight_pct)
    entities = tuple(entity for entity in kg.entities if entity.entity_id in kept_entity_ids)
    if config.lcc_only:
        entities, relations = _largest_connected_component(entities, relations)
    return FastGraphRAGKnowledgeGraph(entities=entities, relations=relations)


def _entity_ids_after_node_filters(
    kg: FastGraphRAGKnowledgeGraph,
    config: FastGraphRAGPruningConfig,
) -> set[str]:
    degree_by_id = _degree_by_entity_id(kg)
    removed_entity_ids: set[str] = set()

    if config.remove_ego_nodes and degree_by_id:
        ego_entity_id = max(degree_by_id, key=lambda entity_id: degree_by_id[entity_id])
        removed_entity_ids.add(ego_entity_id)

    max_degree = _std_ceiling(degree_by_id.values(), config.max_node_degree_std)
    for entity_id, degree in degree_by_id.items():
        if degree < config.min_node_degree:
            removed_entity_ids.add(entity_id)
        if max_degree is not None and degree > max_degree:
            removed_entity_ids.add(entity_id)

    remaining_entities = tuple(
        entity for entity in kg.entities if entity.entity_id not in removed_entity_ids
    )
    freq_by_id = {
        entity.entity_id: len(set(entity.candidate_ids))
        for entity in remaining_entities
    }
    max_freq = _std_ceiling(freq_by_id.values(), config.max_node_freq_std)
    for entity in remaining_entities:
        freq = freq_by_id[entity.entity_id]
        if freq < config.min_node_freq:
            removed_entity_ids.add(entity.entity_id)
        if max_freq is not None and freq > max_freq:
            removed_entity_ids.add(entity.entity_id)

    return {
        entity.entity_id
        for entity in kg.entities
        if entity.entity_id not in removed_entity_ids
    }


def _degree_by_entity_id(kg: FastGraphRAGKnowledgeGraph) -> dict[str, int]:
    degree_by_id = {entity.entity_id: 0 for entity in kg.entities}
    for relation in kg.relations:
        degree_by_id[relation.source_entity_id] = degree_by_id.get(relation.source_entity_id, 0) + 1
        degree_by_id[relation.target_entity_id] = degree_by_id.get(relation.target_entity_id, 0) + 1
    return degree_by_id


def _std_ceiling(values: Iterable[int], factor: float | None) -> float | None:
    if factor is None:
        return None
    numeric = [float(value) for value in values]
    if not numeric:
        return None
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    return mean + math.sqrt(variance) * factor


def _filter_relations_by_weight_percentile(
    relations: tuple[FastGraphRAGRelation, ...],
    percentile: float,
) -> tuple[FastGraphRAGRelation, ...]:
    if not relations or percentile <= 0.0:
        return relations
    threshold = _percentile([relation.weight for relation in relations], percentile)
    return tuple(relation for relation in relations if relation.weight >= threshold)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    pct = min(max(percentile, 0.0), 100.0)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _largest_connected_component(
    entities: tuple[FastGraphRAGEntity, ...],
    relations: tuple[FastGraphRAGRelation, ...],
) -> tuple[tuple[FastGraphRAGEntity, ...], tuple[FastGraphRAGRelation, ...]]:
    entity_ids = {entity.entity_id for entity in entities}
    if not entity_ids:
        return (), ()
    adjacency = {entity_id: set[str]() for entity_id in entity_ids}
    for relation in relations:
        if relation.source_entity_id in entity_ids and relation.target_entity_id in entity_ids:
            adjacency[relation.source_entity_id].add(relation.target_entity_id)
            adjacency[relation.target_entity_id].add(relation.source_entity_id)

    components: list[set[str]] = []
    unvisited = set(entity_ids)
    while unvisited:
        root = min(unvisited)
        component = _collect_component(root, adjacency)
        components.append(component)
        unvisited -= component
    largest = max(components, key=lambda component: (len(component), tuple(sorted(component))))
    kept_entities = tuple(entity for entity in entities if entity.entity_id in largest)
    kept_relations = tuple(
        relation
        for relation in relations
        if relation.source_entity_id in largest and relation.target_entity_id in largest
    )
    return kept_entities, kept_relations


def _collect_component(root: str, adjacency: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    queue: deque[str] = deque([root])
    while queue:
        entity_id = queue.popleft()
        if entity_id in visited:
            continue
        visited.add(entity_id)
        queue.extend(sorted(adjacency[entity_id] - visited))
    return visited


__all__ = ["prune_knowledge_graph"]
