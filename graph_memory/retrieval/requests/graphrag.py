from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from graph_memory.contracts.common import TaskId
from graph_memory.retrieval.requests.text import TextCandidate


@dataclass(frozen=True)
class EntityKnowledgeGraphEntity:
    entity_id: str
    name: str
    normalized_name: str
    normalized_aliases: tuple[str, ...]
    description: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("Entity knowledge graph entity_id must be non-empty.")
        if not self.name.strip() or not self.normalized_name.strip():
            raise ValueError("Entity knowledge graph entity names must be non-empty.")
        if not self.candidate_ids:
            raise ValueError("Entity knowledge graph entity requires candidate mappings.")
        if len(set(self.normalized_aliases)) != len(self.normalized_aliases):
            raise ValueError("Entity knowledge graph aliases must be unique.")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Entity candidate mappings must be unique.")


@dataclass(frozen=True)
class EntityKnowledgeGraphRelation:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    weight: float
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.relation_id.strip():
            raise ValueError("Entity relation_id must be non-empty.")
        if not self.source_entity_id.strip() or not self.target_entity_id.strip():
            raise ValueError("Entity relation endpoints must be non-empty.")
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("Entity relations cannot be self loops.")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("Entity relation weight must be finite and positive.")
        if not self.candidate_ids:
            raise ValueError("Entity relation requires candidate mappings.")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Entity relation candidate mappings must be unique.")


@dataclass(frozen=True)
class EntityKnowledgeGraph:
    entities: tuple[EntityKnowledgeGraphEntity, ...]
    relations: tuple[EntityKnowledgeGraphRelation, ...]

    def __post_init__(self) -> None:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("Entity knowledge graph IDs must be unique.")
        valid_ids = set(entity_ids)
        relation_ids: set[str] = set()
        relation_pairs: set[tuple[str, str]] = set()
        for relation in self.relations:
            if relation.relation_id in relation_ids:
                raise ValueError("Entity knowledge graph relation IDs must be unique.")
            relation_ids.add(relation.relation_id)
            relation_pair = (
                min(relation.source_entity_id, relation.target_entity_id),
                max(relation.source_entity_id, relation.target_entity_id),
            )
            if relation_pair in relation_pairs:
                raise ValueError("Entity knowledge graph relation pairs must be unique.")
            relation_pairs.add(relation_pair)
            if (
                relation.source_entity_id not in valid_ids
                or relation.target_entity_id not in valid_ids
            ):
                raise ValueError("Entity relation references a missing entity.")


@dataclass(frozen=True)
class GraphRAGRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    knowledge_graph: EntityKnowledgeGraph

    def __post_init__(self) -> None:
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("GraphRAG candidate IDs must be unique.")
        valid_candidate_ids = set(candidate_ids)
        referenced_candidate_ids = {
            candidate_id
            for entity in self.knowledge_graph.entities
            for candidate_id in entity.candidate_ids
        } | {
            candidate_id
            for relation in self.knowledge_graph.relations
            for candidate_id in relation.candidate_ids
        }
        missing = sorted(referenced_candidate_ids - valid_candidate_ids)
        if missing:
            raise ValueError(
                f"GraphRAG entity graph references missing candidates: {missing}"
            )


__all__ = [
    "EntityKnowledgeGraph",
    "EntityKnowledgeGraphEntity",
    "EntityKnowledgeGraphRelation",
    "GraphRAGRequest",
]
