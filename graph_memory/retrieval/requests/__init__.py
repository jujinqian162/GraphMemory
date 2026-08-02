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
    ProvenancePathRequest,
    ProvenanceRgcnRequest,
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
    | ProvenancePathRequest
    | ProvenanceRgcnRequest
)

__all__ = [
    "DenseConfigLike",
    "DenseRuntime",
    "EvidenceGraphRankingRequest",
    "ExecutionProvenanceRankingRequest",
    "GraphRAGEntity",
    "GraphRAGKnowledgeGraph",
    "GraphRAGRelation",
    "GraphRAGRequest",
    "GraphRAGTextUnit",
    "JsonScalar",
    "ProvenancePathRequest",
    "ProvenanceRgcnRequest",
    "RankingMethodRequest",
    "TextCandidate",
    "TextRankingRequest",
]
