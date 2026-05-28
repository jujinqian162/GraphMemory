from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from graph_memory.indexes.bm25 import BM25TaskRetriever
from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.text import content_tokens
from graph_memory.rerank import rank_graph_from_initial_scores
from graph_memory.rerank_config import ensure_graph_rerank_config
from graph_memory.retrieval_registry import (
    RetrievalMethodSpec,
    get_method_spec,
)
from graph_memory.types import (
    GraphEdge,
    GraphRerankConfig,
    MemoryGraph,
    MemoryTaskInput,
    RankedNode,
    RankedResult,
    Retriever,
)
from graph_memory.validation import (
    validate_graphs,
    validate_memory_task_inputs,
    validate_ranked_results,
    validate_task_id_alignment,
)

if TYPE_CHECKING:
    from graph_memory.learned.features import SeedSignalProvider, TextEmbeddingProvider


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int = 64, normalize_embeddings: bool = True) -> object:
        ...


class RetrievalMethod(Protocol):
    @property
    def name(self) -> str:
        ...

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        ...


@dataclass(frozen=True)
class RetrievalBuildContext:
    """
    Already-loaded runtime context for constructing one retrieval method.
    构造单个检索方法时使用的已读取运行时上下文。

    Fields / 字段:
    - method: Requested public retrieval method name.
      method：请求的公开检索方法名。
    - task_inputs: Validated task input records used by the method.
      task_inputs：该方法使用的已验证 task input 记录。
    - graphs: Optional graph artifacts for graph-based methods.
      graphs：graph-based method 使用的可选 graph artifact。
    - encoder_model: Frozen dense encoder model name.
      encoder_model：冻结 dense encoder 模型名。
    - query_prefix: Prefix applied to query text before dense encoding.
      query_prefix：dense 编码 query 文本前添加的前缀。
    - passage_prefix: Prefix applied to memory text before dense encoding.
      passage_prefix：dense 编码 memory 文本前添加的前缀。
    - graph_config: Optional graph rerank config object or record.
      graph_config：可选 graph rerank config 对象或 record。
    - dense_encoder: Optional injected dense encoder for tests or cached runtime state.
      dense_encoder：测试或缓存运行状态注入的可选 dense encoder。
    - checkpoint_path: Optional trainable model checkpoint path.
      checkpoint_path：可选可训练模型 checkpoint 路径。
    - text_embedding_provider: Optional injected text embedding provider for trainable methods.
      text_embedding_provider：可训练方法使用的可选文本 embedding provider 注入。
    - seed_signal_provider: Optional injected seed signal provider for trainable methods.
      seed_signal_provider：可训练方法使用的可选 seed signal provider 注入。
    - device: Torch device string for trainable retrieval inference.
      device：可训练检索推理使用的 torch device 字符串。
    """

    method: str
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph] | None
    encoder_model: str
    query_prefix: str
    passage_prefix: str
    graph_config: GraphRerankConfig | Mapping[str, object] | None
    dense_encoder: DenseEncoder | None
    checkpoint_path: str | Path | None
    text_embedding_provider: "TextEmbeddingProvider | None"
    seed_signal_provider: "SeedSignalProvider | None"
    device: str


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


def _build_bm25_method(context: RetrievalBuildContext) -> RetrievalMethod:
    return ScorePipelineMethod(
        name=context.method,
        retriever=_build_seed_retriever(
            method="bm25",
            encoder_model=context.encoder_model,
            query_prefix=context.query_prefix,
            passage_prefix=context.passage_prefix,
            dense_encoder=context.dense_encoder,
        ),
    )


def _build_dense_method(context: RetrievalBuildContext) -> RetrievalMethod:
    return ScorePipelineMethod(
        name=context.method,
        retriever=_build_seed_retriever(
            method="dense",
            encoder_model=context.encoder_model,
            query_prefix=context.query_prefix,
            passage_prefix=context.passage_prefix,
            dense_encoder=context.dense_encoder,
        ),
    )


def _build_graph_rerank_method(context: RetrievalBuildContext) -> RetrievalMethod:
    spec = get_method_spec(context.method)
    if spec.seed_method is None:
        raise ValueError(f"Graph rerank method={context.method} requires a seed method.")
    retriever = _build_seed_retriever(
        method=spec.seed_method,
        encoder_model=context.encoder_model,
        query_prefix=context.query_prefix,
        passage_prefix=context.passage_prefix,
        dense_encoder=context.dense_encoder,
    )
    return _build_graph_rerank_score_pipeline(
        method=context.method,
        retriever=retriever,
        task_inputs=context.task_inputs,
        graphs=context.graphs,
        graph_config=context.graph_config,
    )


def _build_trainable_graph_method(context: RetrievalBuildContext) -> RetrievalMethod:
    if context.checkpoint_path is None:
        raise ValueError(f"Trainable graph method={context.method} requires a checkpoint path.")
    if not context.graphs:
        raise ValueError(f"Trainable graph method={context.method} requires graph inputs.")
    from graph_memory.learned.inference import TrainableGraphRetriever

    return TrainableGraphRetriever.from_checkpoint(
        context.checkpoint_path,
        graphs=context.graphs,
        text_embedding_provider=context.text_embedding_provider,
        seed_signal_provider=context.seed_signal_provider,
        dense_encoder=context.dense_encoder,
        device=context.device,
    )


def _build_method_from_spec(spec: RetrievalMethodSpec, context: RetrievalBuildContext) -> RetrievalMethod:
    if spec.builder_id == "bm25":
        return _build_bm25_method(context)
    if spec.builder_id == "dense":
        return _build_dense_method(context)
    if spec.builder_id == "graph_rerank":
        return _build_graph_rerank_method(context)
    if spec.builder_id == "trainable_graph":
        return _build_trainable_graph_method(context)
    raise ValueError(f"Unsupported retrieval method builder={spec.builder_id} for method={spec.name}.")


def build_retrieval_method(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    encoder_model: str = "intfloat/e5-base-v2",
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None,
    dense_encoder: DenseEncoder | None = None,
    checkpoint_path: str | Path | None = None,
    text_embedding_provider: "TextEmbeddingProvider | None" = None,
    seed_signal_provider: "SeedSignalProvider | None" = None,
    device: str = "cpu",
) -> RetrievalMethod:
    context = RetrievalBuildContext(
        method=method,
        task_inputs=task_inputs,
        graphs=graphs,
        encoder_model=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        graph_config=graph_config,
        dense_encoder=dense_encoder,
        checkpoint_path=checkpoint_path,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=device,
    )
    spec = get_method_spec(method)
    return _build_method_from_spec(spec, context)


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
    graph_config: GraphRerankConfig | Mapping[str, object],
) -> list[RankedResult]:
    spec = get_method_spec(method)
    if not spec.requires_graphs or not spec.requires_graph_config or spec.requires_checkpoint:
        raise ValueError(f"Precomputed graph rerank requires a graph rerank method, got method={method}.")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(task_inputs)

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

    validate_ranked_results(predictions, inputs_by_task_id)
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
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None,
    dense_encoder: DenseEncoder | None = None,
    checkpoint_path: str | Path | None = None,
    text_embedding_provider: "TextEmbeddingProvider | None" = None,
    seed_signal_provider: "SeedSignalProvider | None" = None,
    device: str = "cpu",
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(task_inputs)

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
        checkpoint_path=checkpoint_path,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=device,
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

    validate_ranked_results(predictions, inputs_by_task_id)
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
    graph_config: GraphRerankConfig | Mapping[str, object] | None,
) -> GraphRerankMethod:
    if not graphs:
        raise ValueError(f"Graph rerank method={method} requires graph inputs.")
    rerank_config = ensure_graph_rerank_config(graph_config)
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    validate_graphs(graphs, inputs_by_task_id)
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
    if method == "bm25":
        return BM25TaskRetriever()
    if method == "dense":
        return DenseTaskRetriever(
            model_name=encoder_model,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            encoder=dense_encoder,
        )
    raise ValueError(f"Unsupported retrieval method: {method}")


def _seed_method_for(method: str) -> str:
    spec = get_method_spec(method)
    return spec.seed_method or method

