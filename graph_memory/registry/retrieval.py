from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.registry.ids import StrEnum
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig

if TYPE_CHECKING:
    from graph_memory.embeddings import SentenceEncoder
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.contracts import RetrievalMethod, SeedRanker
    from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
    from graph_memory.retrieval.signals import SeedSignalProvider

PayloadT = TypeVar("PayloadT")


class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    MEMORY_STREAM = "memory_stream"
    DENSE_FT = "dense_ft"
    BM25_GRAPH_RERANK = "bm25_graph_rerank"
    DENSE_GRAPH_RERANK = "dense_graph_rerank"
    DENSE_RGCN_GRAPH_RETRIEVER = "dense_rgcn_graph_retriever"


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
    method: Literal[RetrievalMethodId.DENSE] = RetrievalMethodId.DENSE


@dataclass(frozen=True)
class MemoryStreamRetrievalSettings:
    top_k: int
    encoder: DenseEncoderSettings
    scoring: MemoryStreamScoringConfig = field(
        default_factory=MemoryStreamScoringConfig
    )
    capped_test_count: int | None = None
    method: Literal[RetrievalMethodId.MEMORY_STREAM] = RetrievalMethodId.MEMORY_STREAM

    def __post_init__(self) -> None:
        if self.capped_test_count is not None:
            if isinstance(self.capped_test_count, bool) or not isinstance(self.capped_test_count, int):
                raise ValueError("Memory Stream capped_test_count must be an integer.")
            if self.capped_test_count < 0:
                raise ValueError("Memory Stream capped_test_count must be non-negative.")


def _default_neighbor_type_weights() -> dict[str, float]:
    from graph_memory.retrieval.methods.graph_rerank.config import default_neighbor_type_weights

    return default_neighbor_type_weights()


@dataclass(frozen=True)
class GraphRerankSettings:
    lambda_init: float = 1.0
    lambda_query: float = 0.1
    lambda_neighbor: float = 0.2
    lambda_bridge: float = 0.1
    lambda_path: float = 0.0
    seed_top_s: int = 30
    max_hops: int = 2
    neighbor_type_weights: dict[str, float] = field(default_factory=_default_neighbor_type_weights)


@dataclass(frozen=True)
class SeedRetrievalSettings:
    method: Literal[RetrievalMethodId.BM25, RetrievalMethodId.DENSE]
    encoder: DenseEncoderSettings | None = None


@dataclass(frozen=True)
class GraphRerankRetrievalSettings:
    method: Literal[RetrievalMethodId.BM25_GRAPH_RERANK, RetrievalMethodId.DENSE_GRAPH_RERANK]
    top_k: int
    seed: SeedRetrievalSettings
    rerank: GraphRerankSettings


@dataclass(frozen=True)
class CheckpointGraphRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    method: Literal[RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER] = RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER


@dataclass(frozen=True)
class DenseFinetunedRetrievalSettings:
    top_k: int
    checkpoint: Path
    device: str
    method: Literal[RetrievalMethodId.DENSE_FT] = RetrievalMethodId.DENSE_FT


RetrievalJobSettings: TypeAlias = (
    Bm25RetrievalSettings
    | DenseRetrievalSettings
    | MemoryStreamRetrievalSettings
    | GraphRerankRetrievalSettings
    | CheckpointGraphRetrievalSettings
    | DenseFinetunedRetrievalSettings
)


@dataclass(frozen=True)
class ImportanceArtifactProvenance:
    path: Path
    sha256: str
    schema_version: int


@dataclass(frozen=True)
class RetrievalProvenance:
    method: RetrievalMethodId
    model: Path | None
    device: str | None
    encoder: DenseEncoderSettings | None
    importance: ImportanceArtifactProvenance | None = None


@dataclass(frozen=True)
class BuiltRetrievalMethod:
    method: "RetrievalMethod"
    provenance: RetrievalProvenance


@dataclass(frozen=True)
class SeedRetrieverBuildPayload:
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class FlatRetrievalBuildPayload:
    task_inputs: list[MemoryTaskInput]
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class MemoryStreamBuildPayload:
    task_inputs: list[MemoryTaskInput]
    importance_artifact: "ImportanceArtifact"
    importance_path: Path
    importance_sha256: str
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class GraphRerankBuildPayload:
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph]
    graph_config: object | Mapping[str, object] | None = None
    dense_encoder: "SentenceEncoder | None" = None


@dataclass(frozen=True)
class CheckpointGraphBuildPayload:
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph]
    dense_encoder: "SentenceEncoder | None" = None
    text_embedding_provider: "TextEmbeddingProvider | None" = None
    seed_signal_provider: "SeedSignalProvider | None" = None


def _require_payload(payload: object, expected_type: type[PayloadT], *, method: str) -> PayloadT:
    if isinstance(payload, expected_type):
        return payload
    raise TypeError(f"{method} expected {expected_type.__name__}, got {type(payload).__name__}.")


@dataclass(frozen=True)
class RetrievalBuilderSpec:
    settings_type: type[object]
    build: Callable[[RetrievalJobSettings, object], BuiltRetrievalMethod]


@dataclass(frozen=True)
class RetrievalRegistry:
    builders: Mapping[type[object], RetrievalBuilderSpec]
    seed_build: Callable[[SeedRetrievalSettings, object], "SeedRanker"]

    def build_seed(self, settings: SeedRetrievalSettings, payload: object) -> SeedRanker:
        return self.seed_build(settings, payload)

    def build(self, settings: RetrievalJobSettings, payload: object) -> BuiltRetrievalMethod:
        try:
            spec = self.builders[type(settings)]
        except KeyError as error:
            raise ValueError(f"Unsupported retrieval settings type: {type(settings).__name__}") from error
        return spec.build(settings, payload)

__all__ = [
    "Bm25RetrievalSettings",
    "BuiltRetrievalMethod",
    "CheckpointGraphBuildPayload",
    "CheckpointGraphRetrievalSettings",
    "DenseEncoderSettings",
    "DenseFinetunedRetrievalSettings",
    "DenseRetrievalSettings",
    "FlatRetrievalBuildPayload",
    "GraphRerankBuildPayload",
    "GraphRerankRetrievalSettings",
    "GraphRerankSettings",
    "ImportanceArtifactProvenance",
    "MemoryStreamBuildPayload",
    "MemoryStreamRetrievalSettings",
    "RetrievalBuilderSpec",
    "RetrievalJobSettings",
    "RetrievalMethodId",
    "RetrievalProvenance",
    "RetrievalRegistry",
    "SeedRetrieverBuildPayload",
    "SeedRetrievalSettings",
]
