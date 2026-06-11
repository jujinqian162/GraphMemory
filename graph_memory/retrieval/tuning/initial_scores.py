from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.graphs.index import GraphIndex
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import SeedRetrieverBuildPayload
from graph_memory.registry.retrieval_builders import seed_retrieval_settings_for_method
from graph_memory.retrieval.catalog import get_method_spec
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, ensure_graph_rerank_config
from graph_memory.retrieval.methods.graph_rerank.method import GraphRerankMethod, PrecomputedInitialRetriever
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.validation import (
    validate_graphs,
    validate_memory_task_inputs,
    validate_ranked_results,
    validate_task_id_alignment,
)


@dataclass(frozen=True)
class InitialScoreCache:
    scores_by_task_id: dict[str, dict[str, float]]
    latency_ms_by_task_id: dict[str, float]


def precompute_initial_score_cache(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    dense_runtime: DenseRuntime,
) -> InitialScoreCache:
    seed_retriever = Registry.retrieval.build_seed(
        seed_retrieval_settings_for_method(method=method, dense_config=dense_runtime.config),
        SeedRetrieverBuildPayload(dense_encoder=dense_runtime.encoder),
    )
    scores_by_task_id: dict[str, dict[str, float]] = {}
    latency_ms_by_task_id: dict[str, float] = {}
    for task_input in task_inputs:
        started = time.perf_counter()
        ranked_nodes = seed_retriever.rank(task_input)
        latency_ms_by_task_id[task_input["task_id"]] = (time.perf_counter() - started) * 1000.0
        scores_by_task_id[task_input["task_id"]] = {
            ranked_node.node_id: ranked_node.score for ranked_node in ranked_nodes
        }
    return InitialScoreCache(scores_by_task_id=scores_by_task_id, latency_ms_by_task_id=latency_ms_by_task_id)


def run_graph_rerank_from_initial_score_cache(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph],
    initial_score_cache: InitialScoreCache,
    top_k: int,
    graph_config: GraphRerankConfig | Mapping[str, object],
) -> list[RankedResult]:
    spec = get_method_spec(method)
    if not spec.requires_graphs or not spec.requires_graph_config or spec.requires_checkpoint:
        raise ValueError(f"Precomputed graph rerank requires a graph rerank method, got method={method}.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(task_inputs)

    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    validate_graphs(graphs, inputs_by_task_id)
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(inputs_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    retrieval_method = GraphRerankMethod(
        method,
        PrecomputedInitialRetriever(),
        GraphIndex.from_graphs(graphs),
        ensure_graph_rerank_config(graph_config),
    )
    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        if task_id not in initial_score_cache.scores_by_task_id:
            raise ValueError(f"Missing precomputed initial scores for task_id={task_id}.")
        started = time.perf_counter()
        result = retrieval_method.rank_task_from_scores(
            task_input,
            initial_score_cache.scores_by_task_id[task_id],
            top_k=top_k,
        )
        rerank_latency_ms = (time.perf_counter() - started) * 1000.0
        latency_ms = initial_score_cache.latency_ms_by_task_id.get(task_id, 0.0) + rerank_latency_ms
        predictions.append(
            assemble_ranked_result(
                task_input=task_input,
                method=method,
                ranked_nodes=result.ranked_nodes,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_edges=result.trace.retrieved_edges,
            )
        )

    validate_ranked_results(predictions, inputs_by_task_id)
    return predictions
