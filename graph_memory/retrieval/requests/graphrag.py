from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from graph_memory.contracts.common import TaskId
from graph_memory.retrieval.requests.text import TextCandidate

EntityMentionType = Literal["TITLE_ENTITY", "MENTIONS"]


@dataclass(frozen=True)
class GraphRAGEntityMention:
    candidate_id: str
    entity_id: str
    mention_type: EntityMentionType
    normalized_surface: str
    source_prior: float
    alias_confidence: float
    entity_document_frequency: int
    normalized_idf: float
    mention_confidence: float

    def __post_init__(self) -> None:
        for name in ("candidate_id", "entity_id", "normalized_surface"):
            if not getattr(self, name).strip():
                raise ValueError(f"GraphRAG mention {name} must be non-empty.")
        if self.mention_type not in {"TITLE_ENTITY", "MENTIONS"}:
            raise ValueError("GraphRAG mention_type is unsupported.")
        if self.entity_document_frequency <= 0:
            raise ValueError("GraphRAG entity document frequency must be positive.")
        for name in (
            "source_prior",
            "alias_confidence",
            "normalized_idf",
            "mention_confidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"GraphRAG mention {name} must be in [0, 1].")


@dataclass(frozen=True)
class GraphRAGTitleEntityGroup:
    entity_id: str
    normalized_title_entity: str
    candidate_ids: tuple[str, ...]
    group_size: int
    entity_document_frequency: int
    document_frequency_ratio: float

    def __post_init__(self) -> None:
        if not self.entity_id.strip() or not self.normalized_title_entity.strip():
            raise ValueError("GraphRAG title group entity fields must be non-empty.")
        if not self.candidate_ids or len(self.candidate_ids) != len(
            set(self.candidate_ids)
        ):
            raise ValueError(
                "GraphRAG title group candidates must be non-empty/unique."
            )
        if self.group_size != len(self.candidate_ids):
            raise ValueError("GraphRAG title group size must match candidate IDs.")
        if self.entity_document_frequency < self.group_size:
            raise ValueError("GraphRAG title group document frequency is inconsistent.")
        if not 0.0 < self.document_frequency_ratio <= 1.0:
            raise ValueError("GraphRAG title group DF ratio must be in (0, 1].")


@dataclass(frozen=True)
class GraphRAGResolverEvidence:
    anchor_candidate_id: str
    entity_id: str
    candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    top1_score: float | None
    top2_score: float | None
    score_margin: float | None
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.anchor_candidate_id.strip() or not self.entity_id.strip():
            raise ValueError("GraphRAG resolver evidence IDs must be non-empty.")
        if not self.candidate_ids or len(self.candidate_ids) != len(
            set(self.candidate_ids)
        ):
            raise ValueError(
                "GraphRAG resolver candidate IDs must be non-empty/unique."
            )
        if self.accepted != (self.selected_candidate_id is not None):
            raise ValueError("GraphRAG resolver accepted state must match selection.")
        if self.accepted and self.selected_candidate_id not in self.candidate_ids:
            raise ValueError(
                "GraphRAG resolver selected candidate must belong to group."
            )
        if not self.accepted and not self.rejection_reason:
            raise ValueError("Rejected GraphRAG resolution requires a reason.")


@dataclass(frozen=True)
class GraphRAGCandidateBridge:
    source_candidate_id: str
    target_candidate_id: str
    bridge_entity_id: str
    confidence: float
    resolver_score: float
    resolver_margin: float | None
    construction_reason: str
    direction: Literal["BRIDGE_TO"] = "BRIDGE_TO"

    def __post_init__(self) -> None:
        if self.source_candidate_id == self.target_candidate_id:
            raise ValueError("GraphRAG candidate bridge cannot be a self loop.")
        for name in (
            "source_candidate_id",
            "target_candidate_id",
            "bridge_entity_id",
            "construction_reason",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"GraphRAG bridge {name} must be non-empty.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("GraphRAG bridge confidence must be in [0, 1].")
        if not math.isfinite(self.resolver_score):
            raise ValueError("GraphRAG resolver score must be finite.")
        if self.resolver_margin is not None and not math.isfinite(self.resolver_margin):
            raise ValueError("GraphRAG resolver margin must be finite.")


@dataclass(frozen=True)
class GraphRAGKnowledgeGraph:
    mentions: tuple[GraphRAGEntityMention, ...]
    title_groups: tuple[GraphRAGTitleEntityGroup, ...]

    def __post_init__(self) -> None:
        group_entities = [group.entity_id for group in self.title_groups]
        if len(group_entities) != len(set(group_entities)):
            raise ValueError("GraphRAG title group entity IDs must be unique.")


@dataclass(frozen=True)
class GraphRAGRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    knowledge_graph: GraphRAGKnowledgeGraph

    def __post_init__(self) -> None:
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("GraphRAG candidate IDs must be unique.")
        valid = set(candidate_ids)
        referenced = {mention.candidate_id for mention in self.knowledge_graph.mentions}
        referenced.update(
            candidate_id
            for group in self.knowledge_graph.title_groups
            for candidate_id in group.candidate_ids
        )
        missing = sorted(referenced - valid)
        if missing:
            raise ValueError(
                f"GraphRAG knowledge graph references missing candidates: {missing}"
            )


__all__ = [
    "EntityMentionType",
    "GraphRAGCandidateBridge",
    "GraphRAGEntityMention",
    "GraphRAGKnowledgeGraph",
    "GraphRAGRequest",
    "GraphRAGResolverEvidence",
    "GraphRAGTitleEntityGroup",
]
