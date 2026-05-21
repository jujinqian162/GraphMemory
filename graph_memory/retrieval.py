from __future__ import annotations

import time
from typing import Any

from graph_memory.indexes.bm25 import BM25TaskRetriever
from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.text import content_tokens
from graph_memory.rerank import graph_rerank, induced_retrieved_subgraph
from graph_memory.types import (
    GraphEdge,
    GraphRerankConfig,
    GraphRerankConfigRecord,
    MemoryGraph,
    MemoryTaskInput,
    RankedNode,
    RankedResult,
)
from graph_memory.validation import (
    as_validation_record_map,
    as_validation_records,
    validate_graphs,
    validate_memory_task_inputs,
    validate_ranked_results,
)


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
    dense_encoder: Any | None = None,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(as_validation_records(task_inputs))

    if method in {"bm25", "bm25_graph_rerank"}:
        retriever = BM25TaskRetriever()
    elif method in {"dense", "dense_graph_rerank"}:
        retriever = DenseTaskRetriever(
            model_name=encoder_model,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            encoder=dense_encoder,
        )
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")

    graph_by_task_id: dict[str, MemoryGraph] = {}
    rerank_config = _graph_rerank_config_from_value(graph_config) if _is_graph_method(method) else None
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    if _is_graph_method(method):
        if not graphs:
            raise ValueError(f"Graph rerank method={method} requires graph inputs.")
        validate_graphs(as_validation_records(graphs), as_validation_record_map(inputs_by_task_id))
        graph_by_task_id = {graph["task_id"]: graph for graph in graphs}

    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        started = time.perf_counter()
        initial_ranking = retriever.rank(task_input)
        ranked_nodes = initial_ranking
        retrieved_edges: list[GraphEdge] = []
        if _is_graph_method(method):
            assert rerank_config is not None
            graph = graph_by_task_id[task_input["task_id"]]
            initial_scores = {ranked_node.node_id: ranked_node.score for ranked_node in initial_ranking}
            ranked_nodes = graph_rerank(initial_scores, graph, rerank_config)
            top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
            retrieved_edges = induced_retrieved_subgraph(graph, top_node_ids)["edges"]
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


def _is_graph_method(method: str) -> bool:
    return method in {"bm25_graph_rerank", "dense_graph_rerank"}


def _graph_rerank_config_from_value(value: GraphRerankConfig | GraphRerankConfigRecord | None) -> GraphRerankConfig:
    if value is None:
        raise ValueError("Graph rerank methods require graph_config.")
    if isinstance(value, GraphRerankConfig):
        return value
    return GraphRerankConfig(**value)
