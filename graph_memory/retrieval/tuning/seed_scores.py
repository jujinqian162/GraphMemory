from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.graphs.index import GraphIndex
from graph_memory.registry import Registry
from graph_memory.registry.methods import RetrievalLifecycle
from graph_memory.registry.retrieval import RetrievalMethodId, SeedRetrieverBuildPayload
from graph_memory.registry.retrieval_builders import seed_retrieval_settings_for_method
from graph_memory.retrieval.bulk import BulkSeedRanker, rank_tasks, task_groups
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, ensure_graph_rerank_config
from graph_memory.retrieval.methods.graph_rerank.method import GraphRerankMethod, PrecomputedInitialRetriever
from graph_memory.retrieval.requests import DenseRuntime, GraphRankingRequest, TextRankingRequest
from graph_memory.validation import (
    validate_graphs,
    validate_ranked_results,
    validate_task_id_alignment,
)


@dataclass(frozen=True)
class SeedScoreCache:
    scores_by_task_id: dict[str, dict[str, float]]
    latency_ms_by_task_id: dict[str, float]


def precompute_seed_score_cache(
    *,
    seed_method: RetrievalMethodId,
    ranking_requests: list[TextRankingRequest],
    dense_runtime: DenseRuntime,
) -> SeedScoreCache:
    seed_retriever = Registry.retrieval.build_seed(
        seed_retrieval_settings_for_method(method=seed_method, dense_config=dense_runtime.config),
        SeedRetrieverBuildPayload(dense_encoder=dense_runtime.encoder),
    )
    scores_by_task_id: dict[str, dict[str, float]] = {}
    latency_ms_by_task_id: dict[str, float] = {}
    if isinstance(seed_retriever, BulkSeedRanker):
        for request_group in task_groups(ranking_requests):
            started = time.perf_counter()
            ranked_by_task = rank_tasks(seed_retriever, request_group)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            amortized_latency_ms = elapsed_ms / len(request_group)
            for request, ranked_nodes in zip(request_group, ranked_by_task, strict=True):
                task_id = request.task_id
                latency_ms_by_task_id[task_id] = amortized_latency_ms
                scores_by_task_id[task_id] = {
                    ranked_node.node_id: ranked_node.score for ranked_node in ranked_nodes
                }
        return SeedScoreCache(
            scores_by_task_id=scores_by_task_id,
            latency_ms_by_task_id=latency_ms_by_task_id,
        )

    for request in ranking_requests:
        started = time.perf_counter()
        ranked_nodes = seed_retriever.rank(request)
        latency_ms_by_task_id[request.task_id] = (time.perf_counter() - started) * 1000.0
        scores_by_task_id[request.task_id] = {
            ranked_node.node_id: ranked_node.score for ranked_node in ranked_nodes
        }
    return SeedScoreCache(scores_by_task_id=scores_by_task_id, latency_ms_by_task_id=latency_ms_by_task_id)


def run_graph_rerank_from_seed_score_cache(
    *,
    method: str,
    ranking_requests: list[TextRankingRequest],
    graphs: list[MemoryGraph],
    seed_score_cache: SeedScoreCache,
    top_k: int,
    graph_config: GraphRerankConfig | Mapping[str, object],
) -> list[RankedResult]:
    definition = Registry.methods.get(method)
    if definition.lifecycle is not RetrievalLifecycle.GRAPH_RERANK:
        raise ValueError(f"Precomputed graph rerank requires a graph rerank method, got method={method}.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    requests_by_task_id = {request.task_id: request for request in ranking_requests}
    validate_graphs(graphs, ranking_requests)
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(requests_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    graph_index = GraphIndex.from_graphs(graphs)
    retrieval_method = GraphRerankMethod(
        method,
        PrecomputedInitialRetriever(),
        graph_index,
        ensure_graph_rerank_config(graph_config),
    )
    predictions: list[RankedResult] = []
    for request in ranking_requests:
        task_id = request.task_id
        if task_id not in seed_score_cache.scores_by_task_id:
            raise ValueError(f"Missing precomputed initial scores for task_id={task_id}.")
        graph_request = GraphRankingRequest(
            task_id=task_id,
            query_text=request.query_text,
            candidates=request.candidates,
            graph=graph_index.get_required(task_id),
            initial_scores=seed_score_cache.scores_by_task_id[task_id],
        )
        started = time.perf_counter()
        result = retrieval_method.rank_task_from_scores(
            graph_request,
            top_k=top_k,
        )
        rerank_latency_ms = (time.perf_counter() - started) * 1000.0
        latency_ms = seed_score_cache.latency_ms_by_task_id.get(task_id, 0.0) + rerank_latency_ms
        predictions.append(
            assemble_ranked_result(
                text_request=request,
                method=method,
                ranked_nodes=result.ranked_nodes,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_edges=result.trace.retrieved_edges,
            )
        )

    validate_ranked_results(predictions, ranking_requests)
    return predictions
