from __future__ import annotations

from typing import TypeAlias

from graph_memory.retrieval.requests.evidence_graph import (
    EvidenceGraphRankingRequest,
)
from graph_memory.retrieval.requests.graphrag import (
    EntityMentionType,
    GraphRAGCandidateBridge,
    GraphRAGEntityMention,
    GraphRAGKnowledgeGraph,
    GraphRAGRequest,
    GraphRAGResolverEvidence,
    GraphRAGTitleEntityGroup,
)
from graph_memory.retrieval.requests.provenance_path import ProvenancePathRequest
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
)

__all__ = [
    "DenseConfigLike",
    "DenseRuntime",
    "EntityMentionType",
    "EvidenceGraphRankingRequest",
    "GraphRAGCandidateBridge",
    "GraphRAGEntityMention",
    "GraphRAGKnowledgeGraph",
    "GraphRAGRequest",
    "GraphRAGResolverEvidence",
    "GraphRAGTitleEntityGroup",
    "JsonScalar",
    "ProvenancePathRequest",
    "RankingMethodRequest",
    "TextCandidate",
    "TextRankingRequest",
]
