from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from graph_memory.indexes.bm25 import BM25TaskRetriever
from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.text import content_tokens
from graph_memory.rerank import rank_graph_from_initial_scores
from graph_memory.types import (
    GraphEdge,
    GraphRerankConfig,
    GraphRerankConfigRecord,
    MemoryGraph,
    MemoryTaskInput,
    RankedNode,
    RankedResult,
    Retriever,
    graph_rerank_config_from_value,
)
from graph_memory.validation import (
    as_validation_record_map,
    as_validation_records,
    validate_graphs,
    validate_memory_task_inputs,
    validate_ranked_results,
    validate_task_id_alignment,
)


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int = 64, normalize_embeddings: bool = True) -> object:
        ...


class RetrievalMethod(Protocol):
    name: str

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...


@dataclass(frozen=True)
class InitialScoreCache:
    scores_by_task_id: dict[str, dict[str, float]]
    latency_ms_by_task_id: dict[str, float]


@dataclass(frozen=True)
class ScorePipelineMethod:
    name: str
    retriever: Retriever

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        return self.retriever.rank(task_input), []


@dataclass(frozen=True)
class GraphRerankMethod:
    name: str
    retriever: Retriever
    graph_by_task_id: dict[str, MemoryGraph]
    graph_config: GraphRerankConfig

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        initial_ranking = self.retriever.rank(task_input)
        initial_scores = {ranked_node.node_id: ranked_node.score for ranked_node in initial_ranking}
        return self.rank_task_from_scores(task_input, initial_scores, top_k=top_k)

    def rank_task_from_scores(
        self,
        task_input: MemoryTaskInput,
        initial_scores: dict[str, float],
        *,
        top_k: int,
    ) -> tuple[list[RankedNode], list[GraphEdge]]:
        graph = self.graph_by_task_id.get(task_input["task_id"])
        if graph is None:
            raise ValueError(f"Missing graph for task_id={task_input['task_id']}.")
        result = rank_graph_from_initial_scores(
            initial_scores,
            graph,
            self.graph_config,
            top_k=top_k,
        )
        return result.ranked_nodes, result.retrieved_subgraph["edges"]


class PrecomputedInitialRetriever:
    method_name = "precomputed_initial_scores"

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        raise RuntimeError("Precomputed initial score pipelines require rank_task_from_scores.")


def build_retrieval_method(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    encoder_model: str = "intfloat/e5-base-v2",
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    graph_config: GraphRerankConfig | GraphRerankConfigRecord | None = None,
    dense_encoder: DenseEncoder | None = None,
) -> RetrievalMethod:
    retriever = _build_seed_retriever(
        method=method,
        encoder_model=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        dense_encoder=dense_encoder,
    )

    if method in {"bm25", "dense"}:
        return ScorePipelineMethod(
            name=method,
            retriever=retriever,
        )

    return _build_graph_rerank_score_pipeline(
        method=method,
        retriever=retriever,
        task_inputs=task_inputs,
        graphs=graphs,
        graph_config=graph_config,
    )


def precompute_initial_score_cache(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    encoder_model: str = "intfloat/e5-base-v2",
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    dense_encoder: DenseEncoder | None = None,
) -> InitialScoreCache:
    seed_method = _seed_method_for(method)
    seed_retriever = _build_seed_retriever(
        method=seed_method,
        encoder_model=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        dense_encoder=dense_encoder,
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
    graph_config: GraphRerankConfig | GraphRerankConfigRecord,
) -> list[RankedResult]:
    if method not in {"bm25_graph_rerank", "dense_graph_rerank"}:
        raise ValueError(f"Precomputed graph rerank requires a graph rerank method, got method={method}.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(as_validation_records(task_inputs))

    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    retrieval_method = _build_graph_rerank_score_pipeline(
        method=method,
        retriever=PrecomputedInitialRetriever(),
        task_inputs=task_inputs,
        graphs=graphs,
        graph_config=graph_config,
    )
    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        if task_id not in initial_score_cache.scores_by_task_id:
            raise ValueError(f"Missing precomputed initial scores for task_id={task_id}.")
        started = time.perf_counter()
        ranked_nodes, retrieved_edges = retrieval_method.rank_task_from_scores(
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
                ranked_nodes=ranked_nodes,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_edges=retrieved_edges,
            )
        )

    validate_ranked_results(as_validation_records(predictions), as_validation_record_map(inputs_by_task_id))
    return predictions


def run_retrieval(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    top_k: int,
    encoder_model: str = "intfloat/e5-base-v2",
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    graph_config: GraphRerankConfig | GraphRerankConfigRecord | None = None,
    dense_encoder: DenseEncoder | None = None,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(as_validation_records(task_inputs))

    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    retrieval_method = build_retrieval_method(
        method=method,
        task_inputs=task_inputs,
        graphs=graphs,
        encoder_model=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        graph_config=graph_config,
        dense_encoder=dense_encoder,
    )
    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        started = time.perf_counter()
        ranked_nodes, retrieved_edges = retrieval_method.rank_task(task_input, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        predictions.append(
            assemble_ranked_result(
                task_input=task_input,
                method=method,
                ranked_nodes=ranked_nodes,
                top_k=top_k,
                latency_ms=latency_ms,
                retrieved_edges=retrieved_edges,
            )
        )

    validate_ranked_results(as_validation_records(predictions), as_validation_record_map(inputs_by_task_id))
    return predictions


def assemble_ranked_result(
    *,
    task_input: MemoryTaskInput,
    method: str,
    ranked_nodes: list[RankedNode],
    top_k: int,
    latency_ms: float,
    retrieved_edges: list[GraphEdge],
) -> RankedResult:
    top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
    return {
        "task_id": task_input["task_id"],
        "method": method,
        "ranked_nodes": [
            {"node_id": ranked_node.node_id, "score": ranked_node.score}
            for ranked_node in ranked_nodes
        ],
        "retrieved_subgraph": {
            "nodes": top_node_ids,
            "edges": retrieved_edges,
        },
        "latency_ms": latency_ms,
        "input_tokens": _approx_input_tokens(task_input),
    }


def _approx_input_tokens(task_input: MemoryTaskInput) -> int:
    query_tokens = content_tokens(task_input["query"])
    memory_tokens = [
        token
        for memory_item in task_input["memory_items"]
        for token in content_tokens(f'{memory_item["source"]}. {memory_item["text"]}')
    ]
    return len(query_tokens) + len(memory_tokens)


def _build_graph_rerank_score_pipeline(
    *,
    method: str,
    retriever: Retriever,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    graph_config: GraphRerankConfig | GraphRerankConfigRecord | None,
) -> GraphRerankMethod:
    if not graphs:
        raise ValueError(f"Graph rerank method={method} requires graph inputs.")
    rerank_config = graph_rerank_config_from_value(graph_config)
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    validate_graphs(as_validation_records(graphs), as_validation_record_map(inputs_by_task_id))
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(inputs_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    return GraphRerankMethod(
        name=method,
        retriever=retriever,
        graph_by_task_id={graph["task_id"]: graph for graph in graphs},
        graph_config=rerank_config,
    )


def _build_seed_retriever(
    *,
    method: str,
    encoder_model: str,
    query_prefix: str,
    passage_prefix: str,
    dense_encoder: DenseEncoder | None,
) -> Retriever:
    if method in {"bm25", "bm25_graph_rerank"}:
        return BM25TaskRetriever()
    if method in {"dense", "dense_graph_rerank"}:
        return DenseTaskRetriever(
            model_name=encoder_model,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            encoder=dense_encoder,
        )
    raise ValueError(f"Unsupported retrieval method: {method}")


def _seed_method_for(method: str) -> str:
    if method == "bm25_graph_rerank":
        return "bm25"
    if method == "dense_graph_rerank":
        return "dense"
    if method in {"bm25", "dense"}:
        return method
    raise ValueError(f"Unsupported retrieval method: {method}")

