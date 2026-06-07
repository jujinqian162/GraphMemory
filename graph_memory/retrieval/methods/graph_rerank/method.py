from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.graphs.index import GraphIndex
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, RetrievalTrace, SeedRanker
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.methods.graph_rerank.engine import rank_graph_from_initial_scores


@dataclass(frozen=True)
class GraphRerankMethod:
    name: str
    retriever: SeedRanker
    graphs: GraphIndex
    graph_config: GraphRerankConfig

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> RetrievalMethodResult:
        initial_ranking = self.retriever.rank(task_input)
        initial_scores = {ranked_node.node_id: ranked_node.score for ranked_node in initial_ranking}
        return self.rank_task_from_scores(task_input, initial_scores, top_k=top_k)

    def rank_task_from_scores(
        self,
        task_input: MemoryTaskInput,
        initial_scores: dict[str, float],
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        graph = self.graphs.get_required(task_input["task_id"])
        result = rank_graph_from_initial_scores(
            initial_scores,
            graph,
            self.graph_config,
            top_k=top_k,
        )
        return RetrievalMethodResult(
            ranked_nodes=result.ranked_nodes,
            trace=RetrievalTrace(retrieved_edges=result.retrieved_subgraph["edges"]),
        )


class PrecomputedInitialRetriever:
    method_name = "precomputed_initial_scores"

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        raise RuntimeError("Precomputed initial score pipelines require rank_task_from_scores.")
