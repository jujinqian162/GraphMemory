from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    PositiveInt,
)
from graph_memory.retrieval.requests.text import TextCandidate

EntityMentionType = Literal["TITLE_ENTITY", "MENTIONS"]


class GraphRAGEntityMention(DomainModel):
    candidate_id: NonEmptyStr
    entity_id: NonEmptyStr
    mention_type: EntityMentionType
    normalized_surface: NonEmptyStr
    source_prior: FiniteFloat = Field(ge=0.0, le=1.0)
    alias_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    entity_document_frequency: PositiveInt
    normalized_idf: FiniteFloat = Field(ge=0.0, le=1.0)
    mention_confidence: FiniteFloat = Field(ge=0.0, le=1.0)


class GraphRAGTitleEntityGroup(DomainModel):
    entity_id: NonEmptyStr
    normalized_title_entity: NonEmptyStr
    candidate_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    group_size: PositiveInt
    entity_document_frequency: PositiveInt
    document_frequency_ratio: FiniteFloat = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_group(self) -> "GraphRAGTitleEntityGroup":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("title group candidate IDs must be unique")
        if self.group_size != len(self.candidate_ids):
            raise ValueError("title group size must match candidate IDs")
        if self.entity_document_frequency < self.group_size:
            raise ValueError("title group document frequency is inconsistent")
        return self


class GraphRAGResolverEvidence(DomainModel):
    anchor_candidate_id: NonEmptyStr
    entity_id: NonEmptyStr
    candidate_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    selected_candidate_id: NonEmptyStr | None
    top1_score: FiniteFloat | None
    top2_score: FiniteFloat | None
    score_margin: FiniteFloat | None
    accepted: StrictBool
    rejection_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_resolution(self) -> "GraphRAGResolverEvidence":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("resolver candidate IDs must be unique")
        if self.accepted != (self.selected_candidate_id is not None):
            raise ValueError("resolver accepted state must match selection")
        if self.accepted and self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected candidate must belong to resolver group")
        if not self.accepted and self.rejection_reason is None:
            raise ValueError("rejected resolution requires a reason")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("accepted resolution cannot have a rejection reason")
        return self


class GraphRAGCandidateBridge(DomainModel):
    source_candidate_id: NonEmptyStr
    target_candidate_id: NonEmptyStr
    bridge_entity_id: NonEmptyStr
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    resolver_score: FiniteFloat
    resolver_margin: FiniteFloat | None
    construction_reason: NonEmptyStr
    direction: Literal["BRIDGE_TO"] = "BRIDGE_TO"

    @model_validator(mode="after")
    def _reject_self_loop(self) -> "GraphRAGCandidateBridge":
        if self.source_candidate_id == self.target_candidate_id:
            raise ValueError("candidate bridge cannot be a self loop")
        return self


class GraphRAGKnowledgeGraph(DomainModel):
    mentions: tuple[GraphRAGEntityMention, ...]
    title_groups: tuple[GraphRAGTitleEntityGroup, ...]

    @model_validator(mode="after")
    def _unique_groups(self) -> "GraphRAGKnowledgeGraph":
        group_entities = [group.entity_id for group in self.title_groups]
        if len(group_entities) != len(set(group_entities)):
            raise ValueError("title group entity IDs must be unique")
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
            mention.candidate_id for mention in self.knowledge_graph.mentions
        }
        referenced.update(
            candidate_id
            for group in self.knowledge_graph.title_groups
            for candidate_id in group.candidate_ids
        )
        missing = sorted(referenced - valid)
        if missing:
            raise ValueError(
                f"knowledge graph references missing candidates: {missing}"
            )
        return self


__all__ = [
    "EntityMentionType",
    "GraphRAGCandidateBridge",
    "GraphRAGEntityMention",
    "GraphRAGKnowledgeGraph",
    "GraphRAGRequest",
    "GraphRAGResolverEvidence",
    "GraphRAGTitleEntityGroup",
]
