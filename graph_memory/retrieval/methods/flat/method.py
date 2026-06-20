from __future__ import annotations

from dataclasses import dataclass

from graph_memory.retrieval.contracts import RetrievalMethodResult, SeedRanker
from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest


@dataclass(frozen=True)
class ScorePipelineMethod:
    name: str
    retriever: SeedRanker

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        _ = top_k
        if not isinstance(request, TextRankingRequest):
            raise TypeError(f"{self.name} requires TextRankingRequest, got {type(request).__name__}.")
        return RetrievalMethodResult(ranked_nodes=self.retriever.rank(request))