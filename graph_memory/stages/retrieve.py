from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.selection import temporal_memory_requests_for_dataset, text_ranking_requests_for_dataset
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
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    parse_memory_stream_scoring_config,
)
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest, TextRankingRequest


@dataclass(frozen=True)
class RetrieveStageResult:
    predictions: list[RankedResult]
    provenance: RetrievalProvenance

def run_retrieve_stage(
    config: RetrieveStageConfig,
    *,
    task_inputs: Sequence[object],
    graphs: list[MemoryGraph] | None,
    selected_config: GraphRerankConfig | MemoryStreamScoringConfig | Mapping[str, object] | None = None,
    importance_artifact: ImportanceArtifact | None = None,
    importance_sha256: str | None = None,
    dense_encoder: SentenceEncoder | None = None,
) -> RetrieveStageResult:
    ranking_requests = _text_requests(config, task_inputs)
    temporal_requests = _temporal_requests(config, task_inputs)
    graph_list = graphs or []
    built = Registry.retrieval.build(
        config.job,
        _build_payload(
            config,
            ranking_requests=ranking_requests,
            temporal_requests=temporal_requests,
            graphs=graph_list,
            selected_config=selected_config,
            importance_artifact=importance_artifact,
            importance_sha256=importance_sha256,
            dense_encoder=dense_encoder,
        ),
    )
    predictions = run_retrieval(
        retrieval_method=built.method,
        tasks=built.execution_tasks,
        top_k=config.job.top_k,
    )
    return RetrieveStageResult(predictions=predictions, provenance=built.provenance)

def _build_payload(
    config: RetrieveStageConfig,
    *,
    ranking_requests: list[TextRankingRequest],
    temporal_requests: list[TemporalMemoryRankingRequest],
    graphs: list[MemoryGraph],
    selected_config: GraphRerankConfig | MemoryStreamScoringConfig | Mapping[str, object] | None,
    importance_artifact: ImportanceArtifact | None,
    importance_sha256: str | None,
    dense_encoder: SentenceEncoder | None,
) -> object:
    job = config.job
    if isinstance(job, (Bm25RetrievalSettings, DenseRetrievalSettings, DenseFinetunedRetrievalSettings)):
        return FlatRetrievalBuildPayload(ranking_requests=ranking_requests, dense_encoder=dense_encoder)
    if isinstance(job, MemoryStreamRetrievalSettings):
        return MemoryStreamBuildPayload(
            temporal_requests=temporal_requests,
            scoring_config=_selected_memory_stream_scoring_config(selected_config),
            dense_encoder=dense_encoder,
            importance_artifact=importance_artifact,
            importance_path=config.io.importance,
            importance_sha256=importance_sha256,
        )
    if isinstance(job, GraphRerankRetrievalSettings):
        return GraphRerankBuildPayload(
            ranking_requests=ranking_requests,
            graphs=graphs,
            graph_config=_selected_graph_rerank_config(selected_config),
            dense_encoder=dense_encoder,
        )
    if isinstance(job, CheckpointGraphRetrievalSettings):
        return CheckpointGraphBuildPayload(
            ranking_requests=ranking_requests,
            graphs=graphs,
            dense_encoder=dense_encoder,
        )
    raise ValueError(f"Unsupported retrieval job config: {type(job).__name__}")

def _text_requests(config: RetrieveStageConfig, task_inputs: Sequence[object]) -> list[TextRankingRequest]:
    return text_ranking_requests_for_dataset(config.dataset, task_inputs)

def _temporal_requests(config: RetrieveStageConfig, task_inputs: Sequence[object]) -> list[TemporalMemoryRankingRequest]:
    return temporal_memory_requests_for_dataset(config.dataset, task_inputs)

def _selected_memory_stream_scoring_config(
    selected_config: GraphRerankConfig | MemoryStreamScoringConfig | Mapping[str, object] | None,
) -> MemoryStreamScoringConfig | None:
    if selected_config is None:
        return None
    if isinstance(selected_config, MemoryStreamScoringConfig):
        return selected_config
    if isinstance(selected_config, Mapping):
        return parse_memory_stream_scoring_config(selected_config)
    raise ValueError(
        f"Memory Stream selected config must be MemoryStreamScoringConfig or mapping, got {type(selected_config).__name__}."
    )

def _selected_graph_rerank_config(
    selected_config: GraphRerankConfig | MemoryStreamScoringConfig | Mapping[str, object] | None,
) -> GraphRerankConfig | Mapping[str, object] | None:
    if selected_config is None or isinstance(selected_config, GraphRerankConfig) or isinstance(selected_config, Mapping):
        return selected_config
    raise ValueError(
        f"Graph rerank selected config must be GraphRerankConfig or mapping, got {type(selected_config).__name__}."
    )

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
