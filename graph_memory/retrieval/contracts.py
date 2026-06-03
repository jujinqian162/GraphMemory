from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from graph_memory.contracts.common import NodeId, Score
from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.tasks import MemoryTaskInput


@dataclass(frozen=True)
class RankedNode:
    node_id: NodeId
    score: Score


class Retriever(Protocol):
    @property
    def method_name(self) -> str:
        ...

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        ...


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int = 64, normalize_embeddings: bool = True) -> object:
        ...


class RetrievalMethod(Protocol):
    @property
    def name(self) -> str:
        ...

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...
