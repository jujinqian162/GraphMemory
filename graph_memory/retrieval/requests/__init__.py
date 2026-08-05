from __future__ import annotations

from typing import TypeAlias

from graph_memory.retrieval.requests.evidence_graph import (
    EvidenceGraphRankingRequest,
)
from graph_memory.retrieval.requests.graphrag import (
    GraphRAGEntity,
    GraphRAGKnowledgeGraph,
    GraphRAGRelation,
    GraphRAGRequest,
    GraphRAGTextUnit,
)
from graph_memory.retrieval.requests.provenance_path import (
    ExecutionProvenanceRankingRequest,
)
from graph_memory.retrieval.requests.text import (
    DenseConfigLike,
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
    "EvidenceGraphRankingRequest",
    "ExecutionProvenanceRankingRequest",
    "GraphRAGEntity",
    "GraphRAGKnowledgeGraph",
    "GraphRAGRelation",
    "GraphRAGRequest",
    "GraphRAGTextUnit",
    "JsonScalar",
    "RankingMethodRequest",
    "TextCandidate",
    "TextRankingRequest",
]
