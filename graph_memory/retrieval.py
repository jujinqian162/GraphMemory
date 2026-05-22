from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from graph_memory.indexes.bm25 import BM25TaskRetriever
from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.text import content_tokens
from graph_memory.rerank import (
    bridge_edge_scores,
    expanded_candidate_nodes,
    induced_retrieved_subgraph,
    neighbor_propagation_scores,
    normalize_scores,
    query_overlap_scores,
)
from graph_memory.types import (
    GraphEdge,
    GraphRerankConfig,
    GraphRerankConfigRecord,
    MemoryGraph,
    MemoryTaskInput,
    RankedNode,
    RankedResult,
    Retriever,
)
from graph_memory.validation import (
    as_validation_record_map,
    as_validation_records,
    validate_graphs,
    validate_memory_task_inputs,
    validate_ranked_results,
    validate_task_id_alignment,
)

NormalizationMode = Literal["none", "minmax"]


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int = 64, normalize_embeddings: bool = True) -> object:
        ...


class RetrievalMethod(Protocol):
    name: str

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...


class NodeScoreComponent(Protocol):
    @property
    def weight(self) -> float:
        ...

    @property
    def normalization(self) -> NormalizationMode:
        ...

    def scores(self, context: ScoreContext) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class ScoreContext:
    task_input: MemoryTaskInput
    initial_scores: dict[str, float]
    normalized_initial: dict[str, float]
    graph: MemoryGraph | None = None
    graph_config: GraphRerankConfig | None = None
    candidate_nodes: set[str] | None = None


@dataclass(frozen=True)
class InitialScoreCache:
    scores_by_task_id: dict[str, dict[str, float]]
    latency_ms_by_task_id: dict[str, float]


@dataclass(frozen=True)
class InitialScoreComponent:
    weight: float
    normalization: NormalizationMode

    def scores(self, context: ScoreContext) -> dict[str, float]:
        return context.initial_scores


@dataclass(frozen=True)
class QueryOverlapScoreComponent:
    weight: float
    normalization: NormalizationMode = "none"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        if context.graph is None:
            return {}
        return _filter_candidate_scores(query_overlap_scores(context.graph), context.candidate_nodes)


@dataclass(frozen=True)
class NeighborPropagationScoreComponent:
    weight: float
    normalization: NormalizationMode = "none"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        if context.graph is None or context.graph_config is None:
            return {}
        scores = neighbor_propagation_scores(context.normalized_initial, context.graph, context.graph_config)
        return _filter_candidate_scores(scores, context.candidate_nodes)


@dataclass(frozen=True)
class BridgeScoreComponent:
    weight: float
    normalization: NormalizationMode = "none"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        if context.graph is None or context.graph_config is None:
            return {}
        scores = bridge_edge_scores(context.normalized_initial, context.graph, context.graph_config)
        return _filter_candidate_scores(scores, context.candidate_nodes)


@dataclass(frozen=True)
class ScorePipelineMethod:
    name: str
    retriever: Retriever
    components: Sequence[NodeScoreComponent]
    graph_by_task_id: dict[str, MemoryGraph]
    graph_config: GraphRerankConfig | None = None

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
        normalized_initial = normalize_scores(initial_scores)
        candidate_nodes = (
            expanded_candidate_nodes(normalized_initial, graph, self.graph_config)
            if graph is not None and self.graph_config is not None
            else set(initial_scores)
        )
        context = ScoreContext(
            task_input=task_input,
            initial_scores=initial_scores,
            normalized_initial=normalized_initial,
            graph=graph,
            graph_config=self.graph_config,
            candidate_nodes=candidate_nodes,
        )
        ranked_nodes = _combine_component_scores(initial_scores, context, self.components)
        top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
        retrieved_edges = induced_retrieved_subgraph(graph, top_node_ids)["edges"] if graph is not None else []
        return ranked_nodes, retrieved_edges


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
) -> ScorePipelineMethod:
    if method in {"bm25", "bm25_graph_rerank"}:
        retriever: Retriever = BM25TaskRetriever()
    elif method in {"dense", "dense_graph_rerank"}:
        retriever = DenseTaskRetriever(
            model_name=encoder_model,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            encoder=dense_encoder,
        )
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")

    if method in {"bm25", "dense"}:
        return ScorePipelineMethod(
            name=method,
            retriever=retriever,
            components=[InitialScoreComponent(weight=1.0, normalization="none")],
            graph_by_task_id={},
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
    seed_retrieval_method = build_retrieval_method(
        method=seed_method,
        task_inputs=task_inputs,
        graphs=None,
        encoder_model=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        dense_encoder=dense_encoder,
    )
    scores_by_task_id: dict[str, dict[str, float]] = {}
    latency_ms_by_task_id: dict[str, float] = {}
    for task_input in task_inputs:
        started = time.perf_counter()
        ranked_nodes = seed_retrieval_method.retriever.rank(task_input)
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
) -> ScorePipelineMethod:
    if not graphs:
        raise ValueError(f"Graph rerank method={method} requires graph inputs.")
    rerank_config = _graph_rerank_config_from_value(graph_config)
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    validate_graphs(as_validation_records(graphs), as_validation_record_map(inputs_by_task_id))
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(inputs_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    return ScorePipelineMethod( # score = lambda_init * init + lambda_query * query_overlap...
        name=method,
        retriever=retriever,
        components=_graph_rerank_components(rerank_config),
        graph_by_task_id={graph["task_id"]: graph for graph in graphs},
        graph_config=rerank_config,
    )


def _graph_rerank_components(rerank_config: GraphRerankConfig) -> list[NodeScoreComponent]:
    return [
        InitialScoreComponent(weight=rerank_config.lambda_init, normalization="minmax"),
        QueryOverlapScoreComponent(weight=rerank_config.lambda_query),
        NeighborPropagationScoreComponent(weight=rerank_config.lambda_neighbor),
        BridgeScoreComponent(weight=rerank_config.lambda_bridge),
    ]


def _seed_method_for(method: str) -> str:
    if method == "bm25_graph_rerank":
        return "bm25"
    if method == "dense_graph_rerank":
        return "dense"
    if method in {"bm25", "dense"}:
        return method
    raise ValueError(f"Unsupported retrieval method: {method}")


def _graph_rerank_config_from_value(value: GraphRerankConfig | GraphRerankConfigRecord | None) -> GraphRerankConfig:
    if value is None:
        raise ValueError("Graph rerank methods require graph_config.")
    if isinstance(value, GraphRerankConfig):
        return value
    return GraphRerankConfig(**value)


def _combine_component_scores(
    node_scores: dict[str, float],
    context: ScoreContext,
    components: Sequence[NodeScoreComponent],
) -> list[RankedNode]:
    combined_scores = {node_id: 0.0 for node_id in node_scores}
    for component in components:
        component_scores = _normalize_component_scores(component.scores(context), component.normalization)
        for node_id in combined_scores:
            combined_scores[node_id] += component.weight * component_scores.get(node_id, 0.0)
    ranked_nodes = [
        RankedNode(node_id=node_id, score=score)
        for node_id, score in combined_scores.items()
    ]
    return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))


def _normalize_component_scores(scores: dict[str, float], mode: NormalizationMode) -> dict[str, float]:
    if mode == "none":
        return scores
    return normalize_scores(scores)


def _filter_candidate_scores(scores: dict[str, float], candidate_nodes: set[str] | None) -> dict[str, float]:
    if candidate_nodes is None:
        return scores
    return {node_id: score for node_id, score in scores.items() if node_id in candidate_nodes}
