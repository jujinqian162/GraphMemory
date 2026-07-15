from __future__ import annotations

from typing import TypeAlias

from graph_memory.retrieval.requests.evidence_graph import (
    EvidenceGraphRankingRequest,
)
from graph_memory.retrieval.requests.execution_provenance import (
    ExecutionProvenanceRankingRequest,
)
from graph_memory.retrieval.requests.graphrag import (
    EntityKnowledgeGraph,
    EntityKnowledgeGraphEntity,
    EntityKnowledgeGraphRelation,
    GraphRAGRequest,
)
from graph_memory.retrieval.requests.text import (
    DenseConfigLike,
    DenseRuntime,
    JsonScalar,
    TextCandidate,
    TextRankingRequest,
)

RankingMethodRequest: TypeAlias = (
    TextRankingRequest
    | EvidenceGraphRankingRequest
    | GraphRAGRequest
    | ExecutionProvenanceRankingRequest
)

__all__ = [
    "DenseConfigLike",
    "DenseRuntime",
    "EntityKnowledgeGraph",
    "EntityKnowledgeGraphEntity",
    "EntityKnowledgeGraphRelation",
    "EvidenceGraphRankingRequest",
    "ExecutionProvenanceRankingRequest",
    "GraphRAGRequest",
    "JsonScalar",
    "RankingMethodRequest",
    "TextCandidate",
    "TextRankingRequest",
]
