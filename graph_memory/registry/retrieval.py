from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.compat import StrEnum
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.methods.epgm import (
    EpgmRetrieverConfig,
)
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextRankingRequest,
)

if TYPE_CHECKING:
    from graph_memory.embeddings import SentenceEncoder
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.contracts import RetrievalMethod
    from graph_memory.retrieval.signals import SeedSignalProvider

PayloadT = TypeVar("PayloadT")


class RetrievalTaskFamily(StrEnum):
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    EXECUTION_PROVENANCE = "execution_provenance"


class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    DENSE_FT = "dense_ft"
    GRAPHRAG = "graphrag"
    DENSE_RGCN_GRAPH_RETRIEVER = "dense_rgcn_graph_retriever"
    DENSE_FT_RGCN_GRAPH_RETRIEVER = "dense_ft_rgcn_graph_retriever"
    EXECUTION_PROVENANCE_RETRIEVER = "execution_provenance_retriever"
    EXECUTION_PROVENANCE_RGCN_RETRIEVER = "execution_provenance_rgcn_retriever"


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
    device: str | None = None
    method: Literal[RetrievalMethodId.DENSE] = RetrievalMethodId.DENSE


@dataclass(frozen=True)
class GraphRAGRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    config: GraphRAGConfig = GraphRAGConfig()
    device: str | None = None
    method: Literal[RetrievalMethodId.GRAPHRAG] = RetrievalMethodId.GRAPHRAG


@dataclass(frozen=True)
class SeedRetrievalSettings:
    method: Literal[RetrievalMethodId.BM25, RetrievalMethodId.DENSE]
    encoder: DenseEncoderSettings | None = None
    device: str | None = None


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
class DenseFinetunedRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


@dataclass(frozen=True)
class ExecutionProvenanceRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    config: EpgmRetrieverConfig = EpgmRetrieverConfig()
    device: str | None = None
    method: Literal[RetrievalMethodId.EXECUTION_PROVENANCE_RETRIEVER] = (
        RetrievalMethodId.EXECUTION_PROVENANCE_RETRIEVER
    )


@dataclass(frozen=True)
class ProvenanceRgcnRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    variant: str = "full_rgcn"
    method: Literal[RetrievalMethodId.EXECUTION_PROVENANCE_RGCN_RETRIEVER] = (
        RetrievalMethodId.EXECUTION_PROVENANCE_RGCN_RETRIEVER
    )


RetrievalJobSettings: TypeAlias = (
    Bm25RetrievalSettings
    | DenseRetrievalSettings
    | DenseFinetunedRetrievalSettings
    | GraphRAGRetrievalSettings
    | EvidenceRgcnRetrievalSettings
    | ExecutionProvenanceRetrievalSettings
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
    execution_tasks: list[RetrievalExecutionTask]


@dataclass(frozen=True)
class SeedRetrieverBuildPayload:
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class FlatRetrievalBuildPayload:
    text_requests: list[TextRankingRequest]
    task_family: RetrievalTaskFamily = RetrievalTaskFamily.EVIDENCE_RETRIEVAL
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class GraphRAGBuildPayload:
    text_requests: list[TextRankingRequest]
    task_family: RetrievalTaskFamily = RetrievalTaskFamily.EVIDENCE_RETRIEVAL
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class EvidenceRgcnBuildPayload:
    text_requests: list[TextRankingRequest]
    evidence_graphs: list[EvidenceGraph]
    dense_encoder: "SentenceEncoder | None" = None
    text_embedding_provider: "TextEmbeddingProvider | None" = None
    seed_signal_provider: "SeedSignalProvider | None" = None


@dataclass(frozen=True)
class ExecutionProvenanceBuildPayload:
    provenance_requests: list[ExecutionProvenanceRankingRequest]
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class ProvenanceRgcnBuildPayload:
    provenance_requests: list[ExecutionProvenanceRankingRequest]
    dense_encoder: "SentenceEncoder | None" = None


def _require_payload(
    payload: object,
    expected_type: type[PayloadT],
    *,
    method: str,
) -> PayloadT:
    if isinstance(payload, expected_type):
        return payload
    raise TypeError(
        f"{method} expected {expected_type.__name__}, got {type(payload).__name__}."
    )


@dataclass(frozen=True)
class RetrievalBuilderSpec:
    settings_type: type[object]
    payload_type: type[object]
    build: Callable[[RetrievalJobSettings, object], BuiltRetrievalMethod]


@dataclass(frozen=True)
class RetrievalRegistry:
    builders: Mapping[type[object], RetrievalBuilderSpec]
    validate_request: Callable[
        [str | RetrievalMethodId, object, RetrievalTaskFamily], None
    ]

    def build(
        self, settings: RetrievalJobSettings, payload: object
    ) -> BuiltRetrievalMethod:
        try:
            spec = self.builders[type(settings)]
        except KeyError as error:
            raise ValueError(
                f"Unsupported retrieval settings type: {type(settings).__name__}"
            ) from error
        _require_payload(payload, spec.payload_type, method=settings.method.value)
        built = spec.build(settings, payload)
        family = _payload_family(payload)
        for task in built.execution_tasks:
            self.validate_request(settings.method, task.method_request, family)
        return built


def _payload_family(payload: object) -> RetrievalTaskFamily:
    if isinstance(payload, (FlatRetrievalBuildPayload, GraphRAGBuildPayload)):
        return payload.task_family
    if isinstance(payload, EvidenceRgcnBuildPayload):
        return RetrievalTaskFamily.EVIDENCE_RETRIEVAL
    if isinstance(payload, ExecutionProvenanceBuildPayload):
        return RetrievalTaskFamily.EXECUTION_PROVENANCE
    if isinstance(payload, ProvenanceRgcnBuildPayload):
        return RetrievalTaskFamily.EXECUTION_PROVENANCE
    raise TypeError(f"Unknown retrieval payload type: {type(payload).__name__}.")


__all__ = [
    "Bm25RetrievalSettings",
    "BuiltRetrievalMethod",
    "DenseEncoderSettings",
    "DenseFinetunedRetrievalSettings",
    "DenseRetrievalSettings",
    "EvidenceRgcnBuildPayload",
    "EvidenceRgcnRetrievalSettings",
    "ExecutionProvenanceBuildPayload",
    "ExecutionProvenanceRetrievalSettings",
    "FlatRetrievalBuildPayload",
    "GraphRAGBuildPayload",
    "GraphRAGRetrievalSettings",
    "ProvenanceRgcnBuildPayload",
    "ProvenanceRgcnRetrievalSettings",
    "RetrievalBuilderSpec",
    "RetrievalJobSettings",
    "RetrievalMethodId",
    "RetrievalProvenance",
    "RetrievalRegistry",
    "RetrievalTaskFamily",
    "SeedRetrieverBuildPayload",
    "SeedRetrievalSettings",
]
