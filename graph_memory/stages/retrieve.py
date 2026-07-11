from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.selection import (
    temporal_memory_requests_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.experiment.stage_models import (
    Bm25GraphRerankRetrieveStageConfig,
    Bm25RetrieveStageConfig,
    DenseFinetuneRetrieveStageConfig,
    DenseGraphRerankRetrieveStageConfig,
    DenseRetrieveStageConfig,
    MemoryStreamRetrieveStageConfig,
    RetrieveStageConfig,
    RgcnRetrieveStageConfig,
)
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
    GraphRerankSettings,
    MemoryStreamBuildPayload,
    MemoryStreamRetrievalSettings,
    RetrievalProvenance,
    RetrievalMethodId,
    SeedRetrievalSettings,
)
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.embeddings import SentenceEncoder
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    parse_memory_stream_scoring_config,
)
from graph_memory.retrieval.requests import (
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class RetrieveStageResult:
    predictions: list[RankedResult]
    provenance: RetrievalProvenance


def run_retrieve_stage(
    config: RetrieveStageConfig,
    *,
    task_inputs: Sequence[object],
    graphs: list[MemoryGraph] | None,
    selected_config: GraphRerankConfig
    | MemoryStreamScoringConfig
    | Mapping[str, object]
    | None = None,
    importance_artifact: ImportanceArtifact | None = None,
    importance_sha256: str | None = None,
    dense_encoder: SentenceEncoder | None = None,
) -> RetrieveStageResult:
    ranking_requests = _text_requests(config, task_inputs)
    temporal_requests = _temporal_requests(config, task_inputs)
    graph_list = graphs or []
    settings = _retrieval_settings(config, selected_config)
    built = Registry.retrieval.build(
        settings,
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
        top_k=config.top_k,
    )
    return RetrieveStageResult(predictions=predictions, provenance=built.provenance)


def _build_payload(
    config: RetrieveStageConfig,
    *,
    ranking_requests: list[TextRankingRequest],
    temporal_requests: list[TemporalMemoryRankingRequest],
    graphs: list[MemoryGraph],
    selected_config: GraphRerankConfig
    | MemoryStreamScoringConfig
    | Mapping[str, object]
    | None,
    importance_artifact: ImportanceArtifact | None,
    importance_sha256: str | None,
    dense_encoder: SentenceEncoder | None,
) -> object:
    if isinstance(
        config,
        (
            Bm25RetrieveStageConfig,
            DenseRetrieveStageConfig,
            DenseFinetuneRetrieveStageConfig,
        ),
    ):
        return FlatRetrievalBuildPayload(
            ranking_requests=ranking_requests, dense_encoder=dense_encoder
        )
    if isinstance(config, MemoryStreamRetrieveStageConfig):
        return MemoryStreamBuildPayload(
            temporal_requests=temporal_requests,
            importance_artifact=_require_memory_stream_importance_artifact(
                config, importance_artifact
            ),
            importance_path=_require_memory_stream_importance_path(config),
            importance_sha256=_require_memory_stream_importance_sha256(
                config, importance_sha256
            ),
            scoring_config=_selected_memory_stream_scoring_config(selected_config),
            dense_encoder=dense_encoder,
        )
    if isinstance(
        config,
        (Bm25GraphRerankRetrieveStageConfig, DenseGraphRerankRetrieveStageConfig),
    ):
        return GraphRerankBuildPayload(
            ranking_requests=ranking_requests,
            graphs=graphs,
            graph_config=_selected_graph_rerank_config(selected_config),
            dense_encoder=dense_encoder,
        )
    if isinstance(config, RgcnRetrieveStageConfig):
        return CheckpointGraphBuildPayload(
            ranking_requests=ranking_requests,
            graphs=graphs,
            dense_encoder=dense_encoder,
        )
    raise ValueError(f"Unsupported retrieval stage config: {type(config).__name__}")


def _text_requests(
    config: RetrieveStageConfig, task_inputs: Sequence[object]
) -> list[TextRankingRequest]:
    return text_ranking_requests_for_dataset(config.dataset, task_inputs)


def _temporal_requests(
    config: RetrieveStageConfig, task_inputs: Sequence[object]
) -> list[TemporalMemoryRankingRequest]:
    return temporal_memory_requests_for_dataset(config.dataset, task_inputs)


def _selected_memory_stream_scoring_config(
    selected_config: GraphRerankConfig
    | MemoryStreamScoringConfig
    | Mapping[str, object]
    | None,
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
    selected_config: GraphRerankConfig
    | MemoryStreamScoringConfig
    | Mapping[str, object]
    | None,
) -> GraphRerankConfig | Mapping[str, object] | None:
    if (
        selected_config is None
        or isinstance(selected_config, GraphRerankConfig)
        or isinstance(selected_config, Mapping)
    ):
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
    raise ValueError(
        f"Memory Stream retrieval requires importance artifact: {importance_path}"
    )


def _require_memory_stream_importance_path(config: RetrieveStageConfig) -> Path:
    if not isinstance(config, MemoryStreamRetrieveStageConfig):
        raise ValueError(
            "Memory Stream importance is only available on memory_stream config."
        )
    return config.importance


def _require_memory_stream_importance_sha256(
    config: RetrieveStageConfig,
    importance_sha256: str | None,
) -> str:
    if importance_sha256 is not None:
        return importance_sha256
    importance_path = _require_memory_stream_importance_path(config)
    raise ValueError(
        f"Memory Stream retrieval requires importance SHA-256: {importance_path}"
    )


def _retrieval_settings(
    config: RetrieveStageConfig,
    selected_config: GraphRerankConfig
    | MemoryStreamScoringConfig
    | Mapping[str, object]
    | None,
):
    if isinstance(config, Bm25RetrieveStageConfig):
        return Bm25RetrievalSettings(top_k=config.top_k)
    if isinstance(config, DenseRetrieveStageConfig):
        return DenseRetrievalSettings(
            top_k=config.top_k,
            encoder=_encoder_settings(config.encoder),
        )
    if isinstance(config, MemoryStreamRetrieveStageConfig):
        scoring = _selected_memory_stream_scoring_config(
            selected_config
        ) or MemoryStreamScoringConfig(**config.scoring.model_dump())
        return MemoryStreamRetrievalSettings(
            top_k=config.top_k,
            encoder=_encoder_settings(config.encoder),
            scoring=scoring,
            capped_test_count=config.capped_test_count,
        )
    if isinstance(
        config,
        (Bm25GraphRerankRetrieveStageConfig, DenseGraphRerankRetrieveStageConfig),
    ):
        selected = _selected_graph_rerank_config(selected_config)
        if not isinstance(selected, GraphRerankConfig):
            from graph_memory.retrieval.methods.graph_rerank.config import (
                ensure_graph_rerank_config,
            )

            selected = ensure_graph_rerank_config(selected)
        encoder = (
            _encoder_settings(config.encoder)
            if isinstance(config, DenseGraphRerankRetrieveStageConfig)
            else None
        )
        return GraphRerankRetrievalSettings(
            method=(
                RetrievalMethodId.DENSE_GRAPH_RERANK
                if isinstance(config, DenseGraphRerankRetrieveStageConfig)
                else RetrievalMethodId.BM25_GRAPH_RERANK
            ),
            top_k=config.top_k,
            seed=SeedRetrievalSettings(
                method=(
                    RetrievalMethodId.DENSE
                    if isinstance(config, DenseGraphRerankRetrieveStageConfig)
                    else RetrievalMethodId.BM25
                ),
                encoder=encoder,
            ),
            rerank=GraphRerankSettings(
                lambda_init=selected.lambda_init,
                lambda_query=selected.lambda_query,
                lambda_neighbor=selected.lambda_neighbor,
                lambda_bridge=selected.lambda_bridge,
                lambda_path=selected.lambda_path,
                seed_top_s=selected.seed_top_s,
                max_hops=selected.max_hops,
                neighbor_type_weights=dict(selected.neighbor_type_weights),
            ),
        )
    if isinstance(config, RgcnRetrieveStageConfig):
        return CheckpointGraphRetrievalSettings(
            method=(
                RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER
                if config.method == "dense_ft_rgcn_graph_retriever"
                else RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER
            ),
            top_k=config.top_k,
            checkpoint=config.checkpoint,
            device=config.device,
        )
    if isinstance(config, DenseFinetuneRetrieveStageConfig):
        return DenseFinetunedRetrievalSettings(
            top_k=config.top_k,
            checkpoint=config.model_dir,
            device=config.device,
        )
    raise ValueError(f"Unsupported retrieval stage config: {type(config).__name__}")


def _encoder_settings(config) -> DenseEncoderSettings:
    return DenseEncoderSettings(
        model_name=config.model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


__all__ = ["RetrieveStageResult", "run_retrieve_stage"]
