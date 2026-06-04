from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.graphs.index import GraphIndex
from graph_memory.retrieval.contracts import DenseEncoder
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class DenseRuntime:
    config: DenseConfig
    encoder: DenseEncoder | None = None


@dataclass(frozen=True)
class SeedRetrieverBuildRequest:
    method: str
    dense_runtime: DenseRuntime


@dataclass(frozen=True)
class TrainableGraphRuntime:
    checkpoint_path: str | Path
    device: str
    text_embedding_provider: "TextEmbeddingProvider | None" = None
    seed_signal_provider: "SeedSignalProvider | None" = None
    dense_runtime: DenseRuntime | None = None


@dataclass(frozen=True)
class RetrievalMethodResolveRequest:
    method: str
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph] | None
    dense_runtime: DenseRuntime
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None
    trainable_runtime: TrainableGraphRuntime | None = None


@dataclass(frozen=True)
class FlatMethodBuildRequest:
    method: str
    seed_retriever: SeedRetrieverBuildRequest


@dataclass(frozen=True)
class GraphRerankMethodBuildRequest:
    method: str
    seed_retriever: SeedRetrieverBuildRequest
    graphs: GraphIndex
    config: GraphRerankConfig


@dataclass(frozen=True)
class TrainableGraphMethodBuildRequest:
    method: str
    graphs: GraphIndex
    runtime: TrainableGraphRuntime


MethodBuildRequest: TypeAlias = FlatMethodBuildRequest | GraphRerankMethodBuildRequest | TrainableGraphMethodBuildRequest
