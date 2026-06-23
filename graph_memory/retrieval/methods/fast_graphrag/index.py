from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig
from graph_memory.retrieval.methods.fast_graphrag.edge_weights import pmi_edge_weight
from graph_memory.retrieval.methods.fast_graphrag.nlp import (
    EntityCatalog,
    EntityMention,
    build_entity_catalog,
    extract_candidate_mentions,
)
from graph_memory.retrieval.methods.fast_graphrag.pruning import prune_knowledge_graph
from graph_memory.retrieval.requests import (
    FastGraphRAGEntity,
    FastGraphRAGKnowledgeGraph,
    FastGraphRAGRelation,
    TextRankingRequest,
)
from graph_memory.validation import ContractValidationError, validate_graphs


def build_fast_graphrag_knowledge_graph(
    request: TextRankingRequest,
    graph: MemoryGraph,
    *,
    config: FastGraphRAGConfig | None = None,
) -> FastGraphRAGKnowledgeGraph:
    method_config = config or FastGraphRAGConfig()
    validate_graphs([graph], [request])
    _validate_graph_text_alignment(request, graph)
    catalog = build_entity_catalog(request.candidates, config=method_config.extraction)
    mentions = extract_candidate_mentions(request.candidates, config=method_config.extraction)
    alias_owner = _catalog_alias_owner(catalog)
    entities = tuple(
        sorted(
            (
                FastGraphRAGEntity(
                    entity_id=entity.entity_id,
                    name=entity.name.upper() if entity.entity_type == "noun_phrase" else entity.name,
                    normalized_name=entity.normalized_name,
                    entity_type="NOUN PHRASE" if entity.entity_type == "noun_phrase" else entity.entity_type,
                    description="" if entity.entity_type == "noun_phrase" else entity.description,
                    candidate_ids=entity.candidate_ids,
                )
                for entity in catalog.entities
            ),
            key=lambda entity: entity.entity_id,
        )
    )
    entity_name_by_id = {entity.entity_id: entity.name for entity in entities}
    entity_frequency_by_id = {
        entity.entity_id: len(entity.candidate_ids)
        for entity in entities
    }
    relations = _relations_from_mentions(
        mentions,
        alias_owner,
        entity_name_by_id,
        entity_frequency_by_id,
        normalize_edge_weights=method_config.extraction.normalize_edge_weights,
    )
    raw_kg = FastGraphRAGKnowledgeGraph(entities=entities, relations=relations)
    return prune_knowledge_graph(raw_kg, method_config.pruning)


def _catalog_alias_owner(catalog: EntityCatalog) -> dict[str, str]:
    owners: dict[str, set[str]] = {}
    for entity in catalog.entities:
        for alias in entity.normalized_aliases:
            owners.setdefault(alias, set()).add(entity.entity_id)
    return {
        alias: next(iter(entity_ids))
        for alias, entity_ids in owners.items()
        if len(entity_ids) == 1
    }


def _validate_graph_text_alignment(request: TextRankingRequest, graph: MemoryGraph) -> None:
    node_by_id = {
        str(node["id"]): node
        for node in graph["nodes"]
    }
    query_node = node_by_id.get("q")
    if query_node is None or str(query_node["text"]) != request.query_text:
        raise ContractValidationError(
            f"Invalid FastGraphRAG graph: task_id={request.task_id} graph query text mismatch."
        )
    for candidate in request.candidates:
        node = node_by_id.get(candidate.item_id)
        if node is None or candidate.text not in _candidate_visible_texts(node):
            message = (
                f"Invalid FastGraphRAG graph: task_id={request.task_id} graph node text mismatch for "
                f"candidate_id={candidate.item_id}."
            )
            raise ContractValidationError(message)


def _candidate_visible_texts(node: object) -> set[str]:
    if not isinstance(node, Mapping):
        return set()
    mapping = cast(Mapping[str, object], node)
    raw_text = mapping.get("text")
    if not isinstance(raw_text, str):
        return set()
    texts = {raw_text}
    source_ref = mapping.get("source_ref")
    if isinstance(source_ref, str) and source_ref:
        texts.add(f"{source_ref}. {raw_text}")
    metadata = mapping.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_mapping = cast(Mapping[str, object], metadata)
        title = metadata_mapping.get("title")
        if isinstance(title, str) and title:
            texts.add(f"{title}. {raw_text}")
    return texts


def _relations_from_mentions(
    mentions: Sequence[EntityMention],
    alias_owner: dict[str, str],
    entity_name_by_id: dict[str, str],
    entity_frequency_by_id: dict[str, int],
    *,
    normalize_edge_weights: bool,
) -> tuple[FastGraphRAGRelation, ...]:
    mentions_by_candidate: dict[str, list[EntityMention]] = {}
    for mention in mentions:
        mentions_by_candidate.setdefault(mention.candidate_id, []).append(mention)

    candidate_ids_by_pair: dict[tuple[str, str], set[str]] = {}
    for candidate_id, candidate_mentions in mentions_by_candidate.items():
        canonical_ids = sorted(
            {
                _canonical_entity_id(mention, alias_owner)
                for mention in candidate_mentions
            }
        )
        for index, source_id in enumerate(canonical_ids):
            for target_id in canonical_ids[index + 1 :]:
                first_id, second_id = sorted((source_id, target_id))
                candidate_ids_by_pair.setdefault((first_id, second_id), set()).add(candidate_id)

    total_edge_weights = sum(len(candidate_ids) for candidate_ids in candidate_ids_by_pair.values())
    total_frequency_occurrences = sum(entity_frequency_by_id.values())
    relations: list[FastGraphRAGRelation] = []
    for (source_id, target_id), candidate_ids in sorted(candidate_ids_by_pair.items()):
        count = len(candidate_ids)
        weight = float(count)
        if normalize_edge_weights:
            weight = pmi_edge_weight(
                edge_count=count,
                total_edge_weights=total_edge_weights,
                source_frequency=entity_frequency_by_id.get(source_id, 0),
                target_frequency=entity_frequency_by_id.get(target_id, 0),
                total_frequency_occurrences=total_frequency_occurrences,
            )
        source_name = entity_name_by_id.get(source_id, source_id)
        target_name = entity_name_by_id.get(target_id, target_id)
        relations.append(
            FastGraphRAGRelation(
                relation_id=f"relation:{source_id}:{target_id}",
                source_entity_id=source_id,
                target_entity_id=target_id,
                description=f"{source_name} -- co-occurs with -- {target_name} in {count} text units",
                candidate_ids=tuple(sorted(candidate_ids)),
                weight=weight,
            )
        )
    return tuple(relations)


def _canonical_entity_id(mention: EntityMention, alias_owner: dict[str, str]) -> str:
    return alias_owner.get(mention.normalized_name, mention.entity_id)


def official_noun_graph_snapshot(
    kg: FastGraphRAGKnowledgeGraph,
) -> dict[str, list[dict[str, object]]]:
    entities: list[dict[str, object]] = [
        {
            "title": entity.name,
            "frequency": len(entity.candidate_ids),
            "text_unit_ids": list(entity.candidate_ids),
            "type": entity.entity_type,
            "description": entity.description,
        }
        for entity in kg.entities
    ]
    name_by_id = _entity_name_by_id(kg)
    relationships: list[dict[str, object]] = [
        {
            "source": name_by_id.get(relation.source_entity_id, relation.source_entity_id),
            "target": name_by_id.get(relation.target_entity_id, relation.target_entity_id),
            "weight": relation.weight,
            "text_unit_ids": list(relation.candidate_ids),
            "description": relation.description,
        }
        for relation in kg.relations
    ]
    return {"entities": entities, "relationships": relationships}


def _entity_name_by_id(kg: FastGraphRAGKnowledgeGraph) -> dict[str, str]:
    return {entity.entity_id: entity.name for entity in kg.entities}


__all__ = ["build_fast_graphrag_knowledge_graph", "official_noun_graph_snapshot"]
