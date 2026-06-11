from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphBuildPayload,
    CheckpointGraphRetrievalSettings,
    DenseRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
)
from graph_memory.registry.stage_configs import RetrieveStageConfig
from graph_memory.embeddings import SentenceEncoder
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
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None,
    dense_encoder: SentenceEncoder | None = None,
) -> RetrieveStageResult:
    method = Registry.retrieval.build(
        config.job,
        _build_payload(
            config,
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


def _build_payload(
    config: RetrieveStageConfig,
    *,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    graph_config: GraphRerankConfig | Mapping[str, object] | None,
    dense_encoder: SentenceEncoder | None,
) -> object:
    job = config.job
    if isinstance(job, (Bm25RetrievalSettings, DenseRetrievalSettings)):
        return FlatRetrievalBuildPayload(task_inputs=task_inputs, dense_encoder=dense_encoder)
    if isinstance(job, GraphRerankRetrievalSettings):
        return GraphRerankBuildPayload(
            task_inputs=task_inputs,
            graphs=graphs or [],
            graph_config=graph_config,
            dense_encoder=dense_encoder,
        )
    if isinstance(job, CheckpointGraphRetrievalSettings):
        return CheckpointGraphBuildPayload(
            task_inputs=task_inputs,
            graphs=graphs or [],
            dense_encoder=dense_encoder,
        )
    raise ValueError(f"Unsupported retrieval job config: {type(job).__name__}")


__all__ = ["RetrieveStageResult", "run_retrieve_stage"]
