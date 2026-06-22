from __future__ import annotations

import hashlib
from collections.abc import Sequence

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.retrieval.methods.fast_graphrag.nlp import (
    EntityCatalog,
    EntityMention,
    build_entity_catalog,
    extract_candidate_mentions,
)
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
) -> FastGraphRAGKnowledgeGraph:
    validate_graphs([graph], [request])
    _validate_graph_text_alignment(request, graph)
    catalog = build_entity_catalog(request.candidates)
    mentions = extract_candidate_mentions(request.candidates)
    alias_owner = _catalog_alias_owner(catalog)
    candidate_text_by_id = {candidate.item_id: candidate.text for candidate in request.candidates}
    entities = tuple(
        sorted(
            (
                FastGraphRAGEntity(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    normalized_name=entity.normalized_name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    candidate_ids=entity.candidate_ids,
                )
                for entity in catalog.entities
            ),
            key=lambda entity: entity.entity_id,
        )
    )
    relations = _relations_from_mentions(mentions, alias_owner, candidate_text_by_id)
    return FastGraphRAGKnowledgeGraph(entities=entities, relations=relations)


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
            raise ContractValidationError(
                f"Invalid FastGraphRAG graph: task_id={request.task_id} graph node text mismatch for candidate_id={candidate.item_id}."
            )


def _candidate_visible_texts(node: object) -> set[str]:
    if not isinstance(node, dict):
        return set()
    raw_text = node.get("text")
    if not isinstance(raw_text, str):
        return set()
    texts = {raw_text}
    source_ref = node.get("source_ref")
    if isinstance(source_ref, str) and source_ref:
        texts.add(f"{source_ref}. {raw_text}")
    metadata = node.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if isinstance(title, str) and title:
            texts.add(f"{title}. {raw_text}")
    return texts


def _relations_from_mentions(
    mentions: Sequence[EntityMention],
    alias_owner: dict[str, str],
    candidate_text_by_id: dict[str, str],
) -> tuple[FastGraphRAGRelation, ...]:
    mentions_by_candidate: dict[str, list[EntityMention]] = {}
    for mention in mentions:
        mentions_by_candidate.setdefault(mention.candidate_id, []).append(mention)

    relations: dict[tuple[str, str, str], FastGraphRAGRelation] = {}
    for candidate_id, candidate_mentions in mentions_by_candidate.items():
        title_ids = _candidate_title_entity_ids(candidate_mentions, alias_owner)
        candidate_text = candidate_text_by_id[candidate_id]
        if title_ids:
            for title_id in title_ids:
                for mention in candidate_mentions:
                    target_id = _canonical_entity_id(mention, alias_owner)
                    if target_id == title_id or mention.source in {"title", "alias"}:
                        continue
                    relation = _relation(title_id, target_id, candidate_id, candidate_text, candidate_mentions)
                    relations[(relation.source_entity_id, relation.target_entity_id, relation.relation_id)] = relation
        else:
            canonical_ids = sorted(
                {
                    _canonical_entity_id(mention, alias_owner)
                    for mention in candidate_mentions
                }
            )
            for index, source_id in enumerate(canonical_ids):
                for target_id in canonical_ids[index + 1 :]:
                    relation = _relation(source_id, target_id, candidate_id, candidate_text, candidate_mentions)
                    relations[(relation.source_entity_id, relation.target_entity_id, relation.relation_id)] = relation
    return tuple(sorted(relations.values(), key=lambda relation: relation.relation_id))


def _candidate_title_entity_ids(
    mentions: Sequence[EntityMention],
    alias_owner: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _canonical_entity_id(mention, alias_owner)
                for mention in mentions
                if mention.source == "title"
            }
        )
    )


def _canonical_entity_id(mention: EntityMention, alias_owner: dict[str, str]) -> str:
    return alias_owner.get(mention.normalized_name, mention.entity_id)


def _relation(
    source_id: str,
    target_id: str,
    candidate_id: str,
    candidate_text: str,
    mentions: Sequence[EntityMention],
) -> FastGraphRAGRelation:
    source_name = _entity_name(source_id, mentions)
    target_name = _entity_name(target_id, mentions)
    first_id, second_id = sorted((source_id, target_id))
    relation_id = f"relation:{first_id}:{second_id}:{_candidate_id_hash(candidate_id)}"
    return FastGraphRAGRelation(
        relation_id=relation_id,
        source_entity_id=source_id,
        target_entity_id=target_id,
        description=f"{source_name} -- co-occurs with -- {target_name} in candidate {candidate_id}: {candidate_text}",
        candidate_ids=(candidate_id,),
        weight=1.0,
    )


def _entity_name(entity_id: str, mentions: Sequence[EntityMention]) -> str:
    for mention in sorted(mentions, key=lambda item: (item.source != "title", item.name)):
        if mention.entity_id == entity_id:
            return mention.name
    for mention in sorted(mentions, key=lambda item: item.name):
        if mention.normalized_name and entity_id.endswith(mention.normalized_name.replace(" ", "-")):
            return mention.name
    return entity_id


def _candidate_id_hash(candidate_id: str) -> str:
    return hashlib.sha1(candidate_id.encode("utf-8")).hexdigest()[:12]


__all__ = ["build_fast_graphrag_knowledge_graph"]
