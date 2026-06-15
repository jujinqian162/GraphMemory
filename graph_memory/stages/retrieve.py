from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphBuildPayload,
    CheckpointGraphRetrievalSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    MemoryStreamBuildPayload,
    MemoryStreamRetrievalSettings,
    RetrievalProvenance,
)
from graph_memory.registry.stage_configs import RetrieveStageConfig
from graph_memory.embeddings import SentenceEncoder
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact


@dataclass(frozen=True)
class RetrieveStageResult:
    predictions: list[RankedResult]
    provenance: RetrievalProvenance


def run_retrieve_stage(
    config: RetrieveStageConfig,
    *,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    graph_config: GraphRerankConfig | Mapping[str, object] | None = None,
    importance_artifact: ImportanceArtifact | None = None,
    importance_sha256: str | None = None,
    dense_encoder: SentenceEncoder | None = None,
) -> RetrieveStageResult:
    built = Registry.retrieval.build(
        config.job,
        _build_payload(
            config,
            task_inputs=task_inputs,
            graphs=graphs,
            graph_config=graph_config,
            importance_artifact=importance_artifact,
            importance_sha256=importance_sha256,
            dense_encoder=dense_encoder,
        ),
    )
    predictions = run_retrieval(
        retrieval_method=built.method,
        task_inputs=task_inputs,
        top_k=config.job.top_k,
    )
    return RetrieveStageResult(predictions=predictions, provenance=built.provenance)


def _build_payload(
    config: RetrieveStageConfig,
    *,
    task_inputs: list[MemoryTaskInput],
    graphs: list[MemoryGraph] | None,
    graph_config: GraphRerankConfig | Mapping[str, object] | None,
    importance_artifact: ImportanceArtifact | None,
    importance_sha256: str | None,
    dense_encoder: SentenceEncoder | None,
) -> object:
    job = config.job
    if isinstance(job, (Bm25RetrievalSettings, DenseRetrievalSettings, DenseFinetunedRetrievalSettings)):
        return FlatRetrievalBuildPayload(task_inputs=task_inputs, dense_encoder=dense_encoder)
    if isinstance(job, MemoryStreamRetrievalSettings):
        return MemoryStreamBuildPayload(
            task_inputs=task_inputs,
            importance_artifact=_require_memory_stream_importance_artifact(config, importance_artifact),
            importance_path=_require_memory_stream_importance_path(config),
            importance_sha256=_require_memory_stream_importance_sha256(config, importance_sha256),
            dense_encoder=dense_encoder,
        )
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


def _require_memory_stream_importance_artifact(
    config: RetrieveStageConfig,
    importance_artifact: ImportanceArtifact | None,
) -> ImportanceArtifact:
    if importance_artifact is not None:
        return importance_artifact
    importance_path = _require_memory_stream_importance_path(config)
    raise ValueError(f"Memory Stream retrieval requires importance artifact: {importance_path}")


def _require_memory_stream_importance_path(config: RetrieveStageConfig) -> Path:
    importance_path = config.io.importance
    if importance_path is None:
        raise ValueError("Memory Stream retrieve stage requires RetrieveIO.importance.")
    return importance_path


def _require_memory_stream_importance_sha256(
    config: RetrieveStageConfig,
    importance_sha256: str | None,
) -> str:
    if importance_sha256 is not None:
        return importance_sha256
    importance_path = _require_memory_stream_importance_path(config)
    raise ValueError(f"Memory Stream retrieval requires importance SHA-256: {importance_path}")


__all__ = ["RetrieveStageResult", "run_retrieve_stage"]
