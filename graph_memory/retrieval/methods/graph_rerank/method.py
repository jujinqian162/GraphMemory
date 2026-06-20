from __future__ import annotations

from dataclasses import dataclass

from graph_memory.graphs.index import GraphIndex
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, RetrievalTrace, SeedRanker
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.methods.graph_rerank.engine import rank_graph_from_initial_scores
from graph_memory.retrieval.requests import GraphRankingRequest, RankingMethodRequest, TextRankingRequest


@dataclass(frozen=True)
class GraphRerankMethod:
    name: str
    retriever: SeedRanker
    graphs: GraphIndex
    graph_config: GraphRerankConfig

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        if not isinstance(request, GraphRankingRequest):
            raise TypeError(f"{self.name} requires GraphRankingRequest, got {type(request).__name__}.")
        return self.rank_task_from_scores(request, top_k=top_k)

    def rank_task_from_scores(
        self,
        request: GraphRankingRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        result = rank_graph_from_initial_scores(
            dict(request.initial_scores),
            request.graph,
            self.graph_config,
            top_k=top_k,
        )
        return RetrievalMethodResult(
            ranked_nodes=result.ranked_nodes,
            trace=RetrievalTrace(retrieved_edges=result.retrieved_subgraph["edges"]),
        )


class PrecomputedInitialRetriever:
    method_name = "precomputed_initial_scores"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        raise RuntimeError("Precomputed initial score pipelines require rank_task_from_scores.")