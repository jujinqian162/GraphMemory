from __future__ import annotations

from dataclasses import dataclass

from graph_memory.retrieval.contracts import DenseEncoder
from graph_memory.retrieval.methods.flat.dense import DenseConfig


@dataclass(frozen=True)
class DenseRuntime:
    config: DenseConfig
    encoder: DenseEncoder | None = None
