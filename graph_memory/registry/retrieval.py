from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.registry.ids import StrEnum

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.contracts import DenseEncoder, RetrievalMethod, SeedRanker
    from graph_memory.retrieval.signals import SeedSignalProvider

PayloadT = TypeVar("PayloadT")


class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
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


RetrievalJobSettings: TypeAlias = (
    Bm25RetrievalSettings
    | DenseRetrievalSettings
    | GraphRerankRetrievalSettings
    | CheckpointGraphRetrievalSettings
)


@dataclass(frozen=True)
class RetrievalMethodMetadata:
    name: str
    settings_type: type[object]
    requires_graphs: bool
    requires_graph_config: bool
    requires_checkpoint: bool
    requires_dense_encoder: bool
    seed_method: RetrievalMethodId | None = None


RETRIEVAL_METHOD_METADATA: Mapping[str, RetrievalMethodMetadata] = {
    RetrievalMethodId.BM25.value: RetrievalMethodMetadata(
        name=RetrievalMethodId.BM25.value,
        settings_type=Bm25RetrievalSettings,
        requires_graphs=False,
        requires_graph_config=False,
        requires_checkpoint=False,
        requires_dense_encoder=False,
    ),
    RetrievalMethodId.DENSE.value: RetrievalMethodMetadata(
        name=RetrievalMethodId.DENSE.value,
        settings_type=DenseRetrievalSettings,
        requires_graphs=False,
        requires_graph_config=False,
        requires_checkpoint=False,
        requires_dense_encoder=True,
    ),
    RetrievalMethodId.BM25_GRAPH_RERANK.value: RetrievalMethodMetadata(
        name=RetrievalMethodId.BM25_GRAPH_RERANK.value,
        settings_type=GraphRerankRetrievalSettings,
        requires_graphs=True,
        requires_graph_config=True,
        requires_checkpoint=False,
        requires_dense_encoder=False,
        seed_method=RetrievalMethodId.BM25,
    ),
    RetrievalMethodId.DENSE_GRAPH_RERANK.value: RetrievalMethodMetadata(
        name=RetrievalMethodId.DENSE_GRAPH_RERANK.value,
        settings_type=GraphRerankRetrievalSettings,
        requires_graphs=True,
        requires_graph_config=True,
        requires_checkpoint=False,
        requires_dense_encoder=True,
        seed_method=RetrievalMethodId.DENSE,
    ),
    RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value: RetrievalMethodMetadata(
        name=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value,
        settings_type=CheckpointGraphRetrievalSettings,
        requires_graphs=True,
        requires_graph_config=False,
        requires_checkpoint=True,
        requires_dense_encoder=True,
        seed_method=RetrievalMethodId.DENSE,
    ),
}


@dataclass(frozen=True)
class SeedRetrieverBuildPayload:
    dense_encoder: "DenseEncoder | None" = None


@dataclass(frozen=True)
class FlatRetrievalBuildPayload:
    task_inputs: list[MemoryTaskInput]
    dense_encoder: "DenseEncoder | None" = None


@dataclass(frozen=True)
class GraphRerankBuildPayload:
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph]
    graph_config: object | Mapping[str, object] | None = None
    dense_encoder: "DenseEncoder | None" = None


@dataclass(frozen=True)
class CheckpointGraphBuildPayload:
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph]
    dense_encoder: "DenseEncoder | None" = None
    text_embedding_provider: "TextEmbeddingProvider | None" = None
    seed_signal_provider: "SeedSignalProvider | None" = None


def _require_payload(payload: object, expected_type: type[PayloadT], *, method: str) -> PayloadT:
    if isinstance(payload, expected_type):
        return payload
    raise TypeError(f"{method} expected {expected_type.__name__}, got {type(payload).__name__}.")


@dataclass(frozen=True)
class RetrievalBuilderSpec:
    settings_type: type[object]
    build: Callable[[RetrievalJobSettings, object], "RetrievalMethod"]


@dataclass(frozen=True)
class RetrievalRegistry:
    metadata: Mapping[str, RetrievalMethodMetadata]
    builders: Mapping[type[object], RetrievalBuilderSpec]
    seed_build: Callable[[SeedRetrievalSettings, object], "SeedRanker"]

    def build_seed(self, settings: SeedRetrievalSettings, payload: object) -> SeedRanker:
        return self.seed_build(settings, payload)

    def build(self, settings: RetrievalJobSettings, payload: object) -> RetrievalMethod:
        try:
            spec = self.builders[type(settings)]
        except KeyError as error:
            raise ValueError(f"Unsupported retrieval settings type: {type(settings).__name__}") from error
        return spec.build(settings, payload)


def get_retrieval_method_metadata(method: str) -> RetrievalMethodMetadata:
    try:
        return RETRIEVAL_METHOD_METADATA[method]
    except KeyError as error:
        raise ValueError(f"Unsupported retrieval method: {method}") from error


__all__ = [
    "Bm25RetrievalSettings",
    "CheckpointGraphBuildPayload",
    "CheckpointGraphRetrievalSettings",
    "DenseEncoderSettings",
    "DenseRetrievalSettings",
    "FlatRetrievalBuildPayload",
    "GraphRerankBuildPayload",
    "GraphRerankRetrievalSettings",
    "GraphRerankSettings",
    "RetrievalBuilderSpec",
    "RetrievalJobSettings",
    "RetrievalMethodMetadata",
    "RetrievalMethodId",
    "RetrievalRegistry",
    "RETRIEVAL_METHOD_METADATA",
    "SeedRetrieverBuildPayload",
    "SeedRetrievalSettings",
    "get_retrieval_method_metadata",
]
