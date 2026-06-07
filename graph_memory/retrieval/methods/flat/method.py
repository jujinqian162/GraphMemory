from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RetrievalMethodResult, SeedRanker


@dataclass(frozen=True)
class ScorePipelineMethod:
    name: str
    retriever: SeedRanker

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> RetrievalMethodResult:
        return RetrievalMethodResult(ranked_nodes=self.retriever.rank(task_input))
