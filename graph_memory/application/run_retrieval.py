from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.execution.service import run_retrieval as execute_retrieval
from graph_memory.retrieval.factory import build_retrieval_method
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.requests import DenseRuntime, RetrievalMethodResolveRequest, TrainableGraphRuntime
from graph_memory.retrieval.resolver import resolve_method_build_request


def default_dense_runtime() -> DenseRuntime:
    return DenseRuntime(config=DenseConfig())


@dataclass(frozen=True)
class RunRetrievalRequest:
    """
    Application-level request for one retrieval run.
    一次 retrieval run 的 application 层请求。

    Fields / 字段:
    - method: Public retrieval method name.
      method：公开 retrieval method 名称。
    - task_inputs: Loaded memory task input artifacts.
      task_inputs：已加载的 memory task input artifact。
    - graphs: Optional loaded graph artifacts for graph-backed methods.
      graphs：graph-backed 方法使用的可选 graph artifact。
    - top_k: Number of nodes used for retrieved subgraph assembly.
      top_k：用于 retrieved subgraph 组装的节点数量。
    - dense_runtime: Dense runtime config and optional injected encoder.
      dense_runtime：dense 运行时配置和可选注入 encoder。
    - graph_config: Optional graph-rerank config.
      graph_config：可选 graph-rerank config。
    - trainable_runtime: Optional grouped trainable checkpoint/runtime state.
      trainable_runtime：可选的 trainable checkpoint/runtime 状态组合。
    """

    method: str
    task_inputs: list[MemoryTaskInput]
    graphs: list[MemoryGraph] | None
    top_k: int
    dense_runtime: DenseRuntime = field(default_factory=default_dense_runtime)
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None
    trainable_runtime: TrainableGraphRuntime | None = None


def run_retrieval(request: RunRetrievalRequest) -> list[RankedResult]:
    method_request = resolve_method_build_request(
        RetrievalMethodResolveRequest(
            method=request.method,
            task_inputs=request.task_inputs,
            graphs=request.graphs,
            dense_runtime=request.dense_runtime,
            graph_config=request.graph_config,
            trainable_runtime=request.trainable_runtime,
        )
    )
    retrieval_method = build_retrieval_method(method_request)
    return execute_retrieval(
        retrieval_method=retrieval_method,
        task_inputs=request.task_inputs,
        top_k=request.top_k,
    )
