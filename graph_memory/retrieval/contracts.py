from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from graph_memory.contracts.common import NodeId, Score
from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.tasks import MemoryTaskInput


@dataclass(frozen=True)
class RankedNode:
    node_id: NodeId
    score: Score


@dataclass(frozen=True)
class RetrievalTrace:
    retrieved_edges: list[GraphEdge] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalMethodResult:
    ranked_nodes: list[RankedNode]
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


class SeedRanker(Protocol):
    @property
    def method_name(self) -> str:
        ...

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        ...


class RetrievalMethod(Protocol):
    @property
    def name(self) -> str:
        ...

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> RetrievalMethodResult:
        ...
