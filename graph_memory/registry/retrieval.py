from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.methods.provenance_path import ProvenancePathConfig
from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest

if TYPE_CHECKING:
    from graph_memory.embeddings import SentenceEncoder
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.contracts import RetrievalMethod
    from graph_memory.retrieval.signals import SeedSignalProvider

@dataclass(frozen=True)
class Bm25RetrievalSettings:
    top_k: int
    method: Literal[RetrievalMethodId.BM25] = RetrievalMethodId.BM25


@dataclass(frozen=True)
class DenseEncoderSettings:
    model_name: str
    query_prefix: str
    passage_prefix: str
    batch_size: int = 64


@dataclass(frozen=True)
class DenseRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    device: str
    method: Literal[RetrievalMethodId.DENSE] = RetrievalMethodId.DENSE


@dataclass(frozen=True)
class GraphRAGRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    device: str
    config: GraphRAGConfig = GraphRAGConfig()
    method: Literal[RetrievalMethodId.GRAPHRAG] = RetrievalMethodId.GRAPHRAG


@dataclass(frozen=True)
class ProvenancePathRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    device: str
    config: ProvenancePathConfig = ProvenancePathConfig()
    method: Literal[RetrievalMethodId.PROVENANCE_PATH] = (
        RetrievalMethodId.PROVENANCE_PATH
    )


@dataclass(frozen=True)
class EvidenceRgcnRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    method: Literal[
        RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER,
        RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
    ] = RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER


@dataclass(frozen=True)
class ProvenanceRgcnRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    method: Literal[RetrievalMethodId.PROVENANCE_RGCN] = (
        RetrievalMethodId.PROVENANCE_RGCN
    )


@dataclass(frozen=True)
class DenseFinetunedRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


RetrievalJobSettings: TypeAlias = (
    Bm25RetrievalSettings
    | DenseRetrievalSettings
    | DenseFinetunedRetrievalSettings
    | GraphRAGRetrievalSettings
    | ProvenancePathRetrievalSettings
    | EvidenceRgcnRetrievalSettings
    | ProvenanceRgcnRetrievalSettings
)


@dataclass(frozen=True)
class RetrievalProvenance:
    method: RetrievalMethodId
    model: Path | None
    device: str | None
    encoder: DenseEncoderSettings | None


@dataclass(frozen=True)
class BuiltRetrievalMethod:
    method: "RetrievalMethod"
    provenance: RetrievalProvenance
    execution_requests: list[RankingMethodRequest]


@dataclass(frozen=True)
class FlatRetrievalBuildPayload:
    text_requests: list[TextRankingRequest]
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class GraphRAGBuildPayload:
    text_requests: list[TextRankingRequest]
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class ProvenancePathBuildPayload:
    text_requests: list[TextRankingRequest]
    provenance_graphs: list[ProvenanceGraph]
    graph_ids_by_task_id: Mapping[str, str]
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class ProvenanceRgcnBuildPayload:
    text_requests: list[TextRankingRequest]
    provenance_graphs: list[ProvenanceGraph]
    graph_ids_by_task_id: Mapping[str, str]
    dense_encoder: "SentenceEncoder | None" = None
    text_embedding_provider: "TextEmbeddingProvider | None" = None


@dataclass(frozen=True)
class EvidenceRgcnBuildPayload:
    text_requests: list[TextRankingRequest]
    evidence_graphs: list[EvidenceGraph]
    dense_encoder: "SentenceEncoder | None" = None
    text_embedding_provider: "TextEmbeddingProvider | None" = None
    seed_signal_provider: "SeedSignalProvider | None" = None


__all__ = [
    "Bm25RetrievalSettings",
    "BuiltRetrievalMethod",
    "DenseEncoderSettings",
    "DenseFinetunedRetrievalSettings",
    "DenseRetrievalSettings",
    "EvidenceRgcnBuildPayload",
    "EvidenceRgcnRetrievalSettings",
    "FlatRetrievalBuildPayload",
    "GraphRAGBuildPayload",
    "GraphRAGRetrievalSettings",
    "ProvenancePathBuildPayload",
    "ProvenancePathRetrievalSettings",
    "ProvenanceRgcnBuildPayload",
    "ProvenanceRgcnRetrievalSettings",
    "RetrievalJobSettings",
    "RetrievalMethodId",
    "RetrievalProvenance",
]
