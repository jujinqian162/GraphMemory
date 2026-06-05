from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from graph_memory.retrieval.contracts import DenseEncoder
from graph_memory.retrieval.methods.flat.dense import DenseConfig

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class DenseRuntime:
    config: DenseConfig
    encoder: DenseEncoder | None = None


@dataclass(frozen=True)
class TrainableGraphRuntime:
    checkpoint_path: str | Path
    device: str
    text_embedding_provider: "TextEmbeddingProvider | None" = None
    seed_signal_provider: "SeedSignalProvider | None" = None
    dense_runtime: DenseRuntime | None = None
