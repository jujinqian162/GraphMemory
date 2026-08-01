from __future__ import annotations

from pydantic import Field, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    PositiveFiniteFloat,
    PositiveInt,
)
from graph_memory.retrieval.requests.text import TextCandidate


class GraphRAGTextUnit(DomainModel):
    unit_id: NonEmptyStr
    candidate_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unit(self) -> "GraphRAGTextUnit":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("GraphRAG text-unit candidate IDs must be unique")
        return self


class GraphRAGEntity(DomainModel):
    entity_id: NonEmptyStr
    name: NonEmptyStr
    frequency: PositiveInt
    text_unit_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    candidate_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_entity(self) -> "GraphRAGEntity":
        if len(self.text_unit_ids) != len(set(self.text_unit_ids)):
            raise ValueError("GraphRAG entity text-unit IDs must be unique")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("GraphRAG entity candidate IDs must be unique")
        if self.frequency != len(self.text_unit_ids):
            raise ValueError("GraphRAG entity frequency must equal text-unit frequency")
        return self


class GraphRAGRelation(DomainModel):
    relation_id: NonEmptyStr
    source_entity_id: NonEmptyStr
    target_entity_id: NonEmptyStr
    weight: PositiveFiniteFloat
    text_unit_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    candidate_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_relation(self) -> "GraphRAGRelation":
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("GraphRAG relation cannot be a self loop")
        if len(self.text_unit_ids) != len(set(self.text_unit_ids)):
            raise ValueError("GraphRAG relation text-unit IDs must be unique")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("GraphRAG relation candidate IDs must be unique")
        return self


class GraphRAGKnowledgeGraph(DomainModel):
    text_units: tuple[GraphRAGTextUnit, ...]
    entities: tuple[GraphRAGEntity, ...]
    relations: tuple[GraphRAGRelation, ...]

    @model_validator(mode="after")
    def _validate_graph(self) -> "GraphRAGKnowledgeGraph":
        unit_ids = [unit.unit_id for unit in self.text_units]
        entity_ids = [entity.entity_id for entity in self.entities]
        relation_ids = [relation.relation_id for relation in self.relations]
        for name, values in (
            ("text-unit", unit_ids),
            ("entity", entity_ids),
            ("relation", relation_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"GraphRAG {name} IDs must be unique")
        valid_units = set(unit_ids)
        valid_entities = set(entity_ids)
        relation_pairs: set[tuple[str, str]] = set()
        for entity in self.entities:
            missing = sorted(set(entity.text_unit_ids) - valid_units)
            if missing:
                raise ValueError(
                    f"GraphRAG entity references missing text units={missing}"
                )
        for relation in self.relations:
            if (
                relation.source_entity_id not in valid_entities
                or relation.target_entity_id not in valid_entities
            ):
                raise ValueError("GraphRAG relation references a missing entity")
            missing = sorted(set(relation.text_unit_ids) - valid_units)
            if missing:
                raise ValueError(
                    f"GraphRAG relation references missing text units={missing}"
                )
            source, target = sorted(
                (relation.source_entity_id, relation.target_entity_id)
            )
            pair = (source, target)
            if pair in relation_pairs:
                raise ValueError(f"duplicate GraphRAG relation pair={pair}")
            relation_pairs.add(pair)
        return self


class GraphRAGRequest(DomainModel):
    task_id: NonEmptyStr
    query_text: str
    candidates: tuple[TextCandidate, ...]
    knowledge_graph: GraphRAGKnowledgeGraph

    @model_validator(mode="after")
    def _validate_references(self) -> "GraphRAGRequest":
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("GraphRAG candidate IDs must be unique")
        valid = set(candidate_ids)
        referenced = {
            candidate_id
            for unit in self.knowledge_graph.text_units
            for candidate_id in unit.candidate_ids
        }
        referenced.update(
            candidate_id
            for entity in self.knowledge_graph.entities
            for candidate_id in entity.candidate_ids
        )
        referenced.update(
            candidate_id
            for relation in self.knowledge_graph.relations
            for candidate_id in relation.candidate_ids
        )
        missing = sorted(referenced - valid)
        if missing:
            raise ValueError(f"GraphRAG graph references missing candidates={missing}")
        return self


__all__ = [
    "GraphRAGEntity",
    "GraphRAGKnowledgeGraph",
    "GraphRAGRelation",
    "GraphRAGRequest",
    "GraphRAGTextUnit",
]
