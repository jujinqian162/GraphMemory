from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import RetrievalDependencies
from graph_memory.registry.stage_configs import RetrieveStageConfig
from graph_memory.retrieval.contracts import DenseEncoder
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig


@dataclass(frozen=True)
class RetrieveStageResult:
    predictions: list[RankedResult]


def run_retrieve_stage(
    config: RetrieveStageConfig,
    *,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None, # HUMAN REVIEW POINT: 这个graph和下一行的dense都不是这个上层retriever应该知道的细节啊，抽象层次不对。我再考虑让相应实现层自己通过config.io.graph去读需要的中间文件，而不是交给顶层的run_retrieval去帮忙。
    dense_encoder: DenseEncoder | None = None,
) -> RetrieveStageResult:
    method = Registry.retrieval.build(
        config.job,
        RetrievalDependencies(
            task_inputs=task_inputs,
            graphs=graphs,
            graph_config=graph_config,
            dense_encoder=dense_encoder,
        ),
    )
    predictions = run_retrieval(
        retrieval_method=method,
        task_inputs=task_inputs,
        top_k=config.job.top_k,
    )
    return RetrieveStageResult(predictions=predictions)


__all__ = ["RetrieveStageResult", "run_retrieve_stage"]
