from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RankedNode, Retriever


@dataclass(frozen=True)
class ScorePipelineMethod:
    name: str
    retriever: Retriever

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        return self.retriever.rank(task_input), []
