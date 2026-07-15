from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.selection import (
    execution_provenance_requests_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.embeddings import SentenceEncoder
from graph_memory.experiment.stage_models import (
    Bm25RetrieveStageConfig,
    DenseFinetuneRetrieveStageConfig,
    DenseRetrieveStageConfig,
    ExecutionProvenanceRetrieveStageConfig,
    GraphRAGRetrieveStageConfig,
    RetrieveStageConfig,
    RgcnRetrieveStageConfig,
)
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    DenseEncoderSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    EvidenceRgcnBuildPayload,
    EvidenceRgcnRetrievalSettings,
    ExecutionProvenanceBuildPayload,
    ExecutionProvenanceRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRAGBuildPayload,
    GraphRAGRetrievalSettings,
    RetrievalMethodId,
    RetrievalProvenance,
)
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.graphrag import GraphRAGConfig
from graph_memory.retrieval.methods.execution_provenance import (
    ExecutionProvenanceConfig,
)
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
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
    evidence_graphs: list[EvidenceGraph] | None,
    dense_encoder: SentenceEncoder | None = None,
) -> RetrieveStageResult:
    text_requests = _text_requests(config, task_inputs)
    provenance_requests = _provenance_requests(config, task_inputs)
    settings = _retrieval_settings(config)
    built = Registry.retrieval.build(
        settings,
        _build_payload(
            config,
            text_requests=text_requests,
            evidence_graphs=evidence_graphs or [],
            dense_encoder=dense_encoder,
            provenance_requests=provenance_requests,
        ),
    )
    predictions = run_retrieval(
        retrieval_method=built.method,
        tasks=built.execution_tasks,
        top_k=config.top_k,
    )
    return RetrieveStageResult(
        predictions=predictions,
        provenance=built.provenance,
    )


def _build_payload(
    config: RetrieveStageConfig,
    *,
    text_requests: list[TextRankingRequest],
    evidence_graphs: list[EvidenceGraph],
    dense_encoder: SentenceEncoder | None,
    provenance_requests: list[ExecutionProvenanceRankingRequest],
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
            text_requests=text_requests,
            dense_encoder=dense_encoder,
        )
    if isinstance(config, GraphRAGRetrieveStageConfig):
        return GraphRAGBuildPayload(
            text_requests=text_requests,
            dense_encoder=dense_encoder,
        )
    if isinstance(config, ExecutionProvenanceRetrieveStageConfig):
        return ExecutionProvenanceBuildPayload(
            provenance_requests=provenance_requests,
            dense_encoder=dense_encoder,
        )
    if isinstance(config, RgcnRetrieveStageConfig):
        return EvidenceRgcnBuildPayload(
            text_requests=text_requests,
            evidence_graphs=evidence_graphs,
            dense_encoder=dense_encoder,
        )
    raise ValueError(f"Unsupported retrieval stage config: {type(config).__name__}")


def _text_requests(
    config: RetrieveStageConfig,
    task_inputs: Sequence[object],
) -> list[TextRankingRequest]:
    return text_ranking_requests_for_dataset(config.dataset, task_inputs)


def _provenance_requests(
    config: RetrieveStageConfig,
    task_inputs: Sequence[object],
) -> list[ExecutionProvenanceRankingRequest]:
    if not isinstance(config, ExecutionProvenanceRetrieveStageConfig):
        return []
    return execution_provenance_requests_for_dataset(config.dataset, task_inputs)


def _retrieval_settings(config: RetrieveStageConfig):
    if isinstance(config, Bm25RetrieveStageConfig):
        return Bm25RetrievalSettings(top_k=config.top_k)
    if isinstance(config, DenseRetrieveStageConfig):
        return DenseRetrievalSettings(
            top_k=config.top_k,
            encoder=_encoder_settings(config.encoder),
        )
    if isinstance(config, GraphRAGRetrieveStageConfig):
        return GraphRAGRetrievalSettings(
            top_k=config.top_k,
            encoder=_encoder_settings(config.encoder),
            config=GraphRAGConfig(
                seed_top_s=config.seed_top_s,
                restart_probability=config.restart_probability,
                max_iterations=config.max_iterations,
                convergence_tolerance=config.convergence_tolerance,
                semantic_weight=config.semantic_weight,
                entity_weight=config.entity_weight,
            ),
        )
    if isinstance(config, ExecutionProvenanceRetrieveStageConfig):
        return ExecutionProvenanceRetrievalSettings(
            top_k=config.top_k,
            encoder=_encoder_settings(config.encoder),
            config=ExecutionProvenanceConfig(
                seed_top_s=config.seed_top_s,
                max_hops=config.max_hops,
                top_paths=config.top_paths,
                max_path_expansions=config.max_path_expansions,
                semantic_weight=config.semantic_weight,
                dependency_weight=config.dependency_weight,
                binding_weight=config.binding_weight,
                grounding_weight=config.grounding_weight,
                hop_penalty=config.hop_penalty,
                invalidation_penalty=config.invalidation_penalty,
            ),
        )
    if isinstance(config, RgcnRetrieveStageConfig):
        return EvidenceRgcnRetrievalSettings(
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
