from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from graph_memory.embeddings import SentenceEncoder, load_sentence_transformer
from graph_memory.graphs.index import GraphIndex
from graph_memory.models.dense_finetune.metadata import load_dense_ft_model_metadata
from graph_memory.registry.methods import MethodRegistry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    BuiltRetrievalMethod,
    DenseEncoderSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    EvidenceRgcnBuildPayload,
    EvidenceRgcnRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRAGBuildPayload,
    GraphRAGRetrievalSettings,
    ProvenancePathBuildPayload,
    ProvenancePathRetrievalSettings,
    ProvenanceRgcnBuildPayload,
    ProvenanceRgcnRetrievalSettings,
    RetrievalBuilderSpec,
    RetrievalMethodId,
    RetrievalProvenance,
    RetrievalRegistry,
    SeedRetrievalSettings,
)
from graph_memory.retrieval.contracts import RetrievalMethod
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.retrieval.methods.graphrag import (
    GraphRAGMethod,
    build_graphrag_knowledge_graph,
    build_graphrag_request,
)
from graph_memory.retrieval.requests import (
    DenseConfigLike,
    EvidenceGraphRankingRequest,
    GraphRAGKnowledgeGraph,
    ProvenancePathRequest,
    ProvenanceRgcnRequest,
    TextRankingRequest,
)
from graph_memory.retrieval.signals import SeedSignalProvider


def build_retrieval_registry(method_registry: MethodRegistry) -> RetrievalRegistry:
    return RetrievalRegistry(
        validate_request=method_registry.validate_request,
        builders={
            Bm25RetrievalSettings: RetrievalBuilderSpec(
                Bm25RetrievalSettings,
                FlatRetrievalBuildPayload,
                lambda settings, deps: _build_bm25(
                    cast(Bm25RetrievalSettings, settings), deps
                ),
            ),
            DenseRetrievalSettings: RetrievalBuilderSpec(
                DenseRetrievalSettings,
                FlatRetrievalBuildPayload,
                lambda settings, deps: _build_dense(
                    cast(DenseRetrievalSettings, settings), deps
                ),
            ),
            DenseFinetunedRetrievalSettings: RetrievalBuilderSpec(
                DenseFinetunedRetrievalSettings,
                FlatRetrievalBuildPayload,
                lambda settings, deps: _build_dense_ft(
                    cast(DenseFinetunedRetrievalSettings, settings), deps
                ),
            ),
            GraphRAGRetrievalSettings: RetrievalBuilderSpec(
                GraphRAGRetrievalSettings,
                GraphRAGBuildPayload,
                lambda settings, deps: _build_graphrag(
                    cast(GraphRAGRetrievalSettings, settings), deps
                ),
            ),
            ProvenancePathRetrievalSettings: RetrievalBuilderSpec(
                ProvenancePathRetrievalSettings,
                ProvenancePathBuildPayload,
                lambda settings, deps: _build_provenance_path(
                    cast(ProvenancePathRetrievalSettings, settings), deps
                ),
            ),
            ProvenanceRgcnRetrievalSettings: RetrievalBuilderSpec(
                ProvenanceRgcnRetrievalSettings,
                ProvenanceRgcnBuildPayload,
                lambda settings, deps: _build_provenance_rgcn(
                    cast(ProvenanceRgcnRetrievalSettings, settings), deps
                ),
            ),
            EvidenceRgcnRetrievalSettings: RetrievalBuilderSpec(
                EvidenceRgcnRetrievalSettings,
                EvidenceRgcnBuildPayload,
                lambda settings, deps: _build_evidence_rgcn(
                    cast(EvidenceRgcnRetrievalSettings, settings), deps
                ),
            ),
        },
    )


def seed_retrieval_settings_for_method(
    *,
    method: RetrievalMethodId,
    device: str,
    dense_config: DenseConfigLike | None = None,
) -> SeedRetrievalSettings:
    if method is RetrievalMethodId.BM25:
        return SeedRetrievalSettings(
            method=RetrievalMethodId.BM25,
            device=device,
        )
    if method is RetrievalMethodId.DENSE:
        return SeedRetrievalSettings(
            method=RetrievalMethodId.DENSE,
            encoder=_dense_encoder_settings(dense_config),
            device=device,
        )
    raise ValueError(f"Unsupported seed retrieval method: {method.value}")


def _dense_encoder_settings(config: DenseConfigLike | None) -> DenseEncoderSettings:
    if config is None:
        return DenseEncoderSettings(
            model_name="intfloat/e5-base-v2",
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=64,
        )
    return DenseEncoderSettings(
        model_name=config.model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


def _build_bm25(
    settings: Bm25RetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    # Payload type already checked by RetrievalRegistry.build.
    build_payload = cast(FlatRetrievalBuildPayload, payload)
    return _built(
        ScorePipelineMethod(name=settings.method.value, retriever=BM25TaskRetriever()),
        method=settings.method,
        execution_tasks=_text_execution_tasks(build_payload.text_requests),
    )


def _build_dense(
    settings: DenseRetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    build_payload = cast(FlatRetrievalBuildPayload, payload)
    return _built(
        ScorePipelineMethod(
            name=settings.method.value,
            retriever=DenseTaskRetriever(
                config=DenseConfig(
                    model_name=settings.encoder.model_name,
                    query_prefix=settings.encoder.query_prefix,
                    passage_prefix=settings.encoder.passage_prefix,
                    batch_size=settings.encoder.batch_size,
                    device=settings.device,
                ),
                encoder=build_payload.dense_encoder,
                device=settings.device,
            ),
        ),
        method=settings.method,
        device=settings.device,
        encoder=settings.encoder,
        execution_tasks=_text_execution_tasks(build_payload.text_requests),
    )


def _build_dense_ft(
    settings: DenseFinetunedRetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    build_payload = cast(FlatRetrievalBuildPayload, payload)
    metadata = load_dense_ft_model_metadata(settings.checkpoint)
    encoder = build_payload.dense_encoder
    if encoder is None:
        try:
            encoder = cast(
                SentenceEncoder,
                cast(
                    object,
                    load_sentence_transformer(
                        settings.checkpoint, device=settings.device
                    ),
                ),
            )
        except RuntimeError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense-ft retrieval."
            ) from error
    method = ScorePipelineMethod(
        name=settings.method.value,
        retriever=DenseTaskRetriever(
            config=DenseConfig(
                device=settings.device,
                model_name=str(settings.checkpoint),
                query_prefix=metadata.query_prefix,
                passage_prefix=metadata.passage_prefix,
                batch_size=metadata.batch_size,
            ),
            encoder=encoder,
            device=settings.device,
        ),
    )
    return _built(
        method,
        method=settings.method,
        model=settings.checkpoint,
        device=settings.device,
        encoder=DenseEncoderSettings(
            model_name=metadata.base_model,
            query_prefix=metadata.query_prefix,
            passage_prefix=metadata.passage_prefix,
            batch_size=metadata.batch_size,
        ),
        execution_tasks=_text_execution_tasks(build_payload.text_requests),
    )


def _build_graphrag(
    settings: GraphRAGRetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    build_payload = cast(GraphRAGBuildPayload, payload)
    dense_ranker = _build_dense_ranker(
        settings.encoder, build_payload.dense_encoder, device=settings.device
    )
    graph_by_candidate_ids: dict[tuple[str, ...], GraphRAGKnowledgeGraph] = {}
    execution_tasks: list[RetrievalExecutionTask] = []
    for request in build_payload.text_requests:
        candidate_ids = tuple(candidate.item_id for candidate in request.candidates)
        graph = graph_by_candidate_ids.get(candidate_ids)
        if graph is None:
            graph = build_graphrag_knowledge_graph(
                request.candidates,
                config=settings.config,
            )
            graph_by_candidate_ids[candidate_ids] = graph
        execution_tasks.append(
            RetrievalExecutionTask(
                text_request=request,
                method_request=build_graphrag_request(
                    request,
                    settings.config,
                    knowledge_graph=graph,
                ),
            )
        )
    return _built(
        GraphRAGMethod(
            dense_ranker=dense_ranker,
            config=settings.config,
        ),
        method=settings.method,
        device=settings.device,
        encoder=settings.encoder,
        execution_tasks=execution_tasks,
    )


def _build_provenance_path(
    settings: ProvenancePathRetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.provenance_path import ProvenancePathMethod

    build_payload = cast(ProvenancePathBuildPayload, payload)
    dense_ranker = _build_dense_ranker(
        settings.encoder,
        build_payload.dense_encoder,
        device=settings.device,
    )
    graph_by_id = {graph.graph_id: graph for graph in build_payload.provenance_graphs}
    if len(graph_by_id) != len(build_payload.provenance_graphs):
        raise ValueError("provenance path graph IDs must be unique")
    request_ids = {request.task_id for request in build_payload.text_requests}
    if set(build_payload.graph_ids_by_task_id) != request_ids:
        raise ValueError("provenance path task-to-graph bindings must cover requests")
    execution_tasks: list[RetrievalExecutionTask] = []
    for request in build_payload.text_requests:
        graph_id = build_payload.graph_ids_by_task_id[request.task_id]
        try:
            graph = graph_by_id[graph_id]
        except KeyError as error:
            raise ValueError(
                f"provenance path task={request.task_id} references "
                f"missing graph={graph_id}"
            ) from error
        execution_tasks.append(
            RetrievalExecutionTask(
                text_request=request,
                method_request=ProvenancePathRequest(
                    task_id=request.task_id,
                    query_text=request.query_text,
                    candidates=request.candidates,
                    graph=graph,
                ),
            )
        )
    return _built(
        ProvenancePathMethod(
            dense_ranker=dense_ranker,
            config=settings.config,
        ),
        method=settings.method,
        device=settings.device,
        encoder=settings.encoder,
        execution_tasks=execution_tasks,
    )


def _build_provenance_rgcn(
    settings: ProvenanceRgcnRetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
    from graph_memory.models.graph_retriever.text_embeddings import (
        DenseGraphFeatureProvider,
    )
    from graph_memory.retrieval.methods.trainable_graph import (
        ProvenanceRgcnRetrievalMethod,
    )

    build_payload = cast(ProvenanceRgcnBuildPayload, payload)
    checkpoint = load_rgcn_checkpoint(
        settings.checkpoint,
        expected_method=RetrievalMethodId.PROVENANCE_RGCN,
        map_location=settings.device,
    )
    provider = build_payload.text_embedding_provider
    if provider is None:
        provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            batch_size=checkpoint.model_config.encoder_batch_size,
            device=settings.device,
            encoder=cast(SentenceEncoder | None, build_payload.dense_encoder),
        )
    method = ProvenanceRgcnRetrievalMethod.from_checkpoint(
        settings.checkpoint,
        text_embedding_provider=provider,
        device=settings.device,
    )
    graph_by_id = {graph.graph_id: graph for graph in build_payload.provenance_graphs}
    if len(graph_by_id) != len(build_payload.provenance_graphs):
        raise ValueError("provenance R-GCN graph IDs must be unique")
    request_ids = {request.task_id for request in build_payload.text_requests}
    if set(build_payload.graph_ids_by_task_id) != request_ids:
        raise ValueError("provenance R-GCN task-to-graph bindings must cover requests")
    execution_tasks: list[RetrievalExecutionTask] = []
    for request in build_payload.text_requests:
        graph_id = build_payload.graph_ids_by_task_id[request.task_id]
        try:
            graph = graph_by_id[graph_id]
        except KeyError as error:
            raise ValueError(
                f"provenance R-GCN task={request.task_id} references "
                f"missing graph={graph_id}"
            ) from error
        execution_tasks.append(
            RetrievalExecutionTask(
                text_request=request,
                method_request=ProvenanceRgcnRequest(
                    task_id=request.task_id,
                    query_text=request.query_text,
                    candidates=request.candidates,
                    graph=graph,
                ),
            )
        )
    return _built(
        method,
        method=settings.method,
        model=settings.checkpoint,
        device=settings.device,
        encoder=DenseEncoderSettings(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            batch_size=checkpoint.model_config.encoder_batch_size,
        ),
        execution_tasks=execution_tasks,
    )


def _build_evidence_rgcn(
    settings: EvidenceRgcnRetrievalSettings,
    payload: object,
) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.trainable_graph import (
        TrainableGraphRetrievalMethod,
    )

    build_payload = cast(EvidenceRgcnBuildPayload, payload)
    graph_index = GraphIndex.from_graphs(build_payload.evidence_graphs)
    text_embedding_provider, seed_signal_provider, checkpoint = (
        _evidence_rgcn_providers(settings, build_payload)
    )
    method = TrainableGraphRetrievalMethod.from_checkpoint(
        settings.checkpoint,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=settings.device,
        expected_method=settings.method,
    )
    return _built(
        method,
        method=settings.method,
        model=settings.checkpoint,
        device=settings.device,
        encoder=DenseEncoderSettings(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            batch_size=checkpoint.model_config.encoder_batch_size,
        ),
        execution_tasks=_evidence_execution_tasks(
            build_payload.text_requests,
            graph_index,
            _initial_scores_from_seed_signal_provider(seed_signal_provider),
        ),
    )



def _evidence_rgcn_providers(
    settings: EvidenceRgcnRetrievalSettings,
    payload: EvidenceRgcnBuildPayload,
):
    from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
    from graph_memory.models.graph_retriever.text_embeddings import (
        DenseGraphFeatureProvider,
    )
    from graph_memory.retrieval.signals import RetrieverSeedSignalProvider

    checkpoint = load_rgcn_checkpoint(
        settings.checkpoint,
        expected_method=settings.method,
        map_location=settings.device,
    )

    if (
        payload.text_embedding_provider is not None
        and payload.seed_signal_provider is not None
    ):
        return (
            payload.text_embedding_provider,
            payload.seed_signal_provider,
            checkpoint,
        )
    if payload.text_embedding_provider is None and payload.seed_signal_provider is None:
        joint_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            device=settings.device,
            encoder=cast(SentenceEncoder | None, payload.dense_encoder),
        )
        return joint_provider, joint_provider, checkpoint

    text_embedding_provider = payload.text_embedding_provider
    if text_embedding_provider is None:
        text_embedding_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            device=settings.device,
            encoder=cast(SentenceEncoder | None, payload.dense_encoder),
        )

    seed_signal_provider = payload.seed_signal_provider
    if seed_signal_provider is None:
        encoder = getattr(
            text_embedding_provider,
            "encoder",
            payload.dense_encoder,
        )
        seed_signal_provider = RetrieverSeedSignalProvider(
            DenseTaskRetriever(
                model_name=checkpoint.model_config.encoder_model,
                query_prefix=checkpoint.model_config.query_prefix,
                passage_prefix=checkpoint.model_config.passage_prefix,
                encoder=cast(SentenceEncoder | None, encoder),
                device=settings.device,
            )
        )
    return text_embedding_provider, seed_signal_provider, checkpoint


def _resolve_encoder(
    settings: DenseEncoderSettings,
    encoder: SentenceEncoder | None,
    *,
    device: str,
) -> SentenceEncoder:
    if encoder is not None:
        return encoder
    try:
        return cast(
            SentenceEncoder,
            cast(
                object,
                load_sentence_transformer(settings.model_name, device=device),
            ),
        )
    except RuntimeError as error:
        raise RuntimeError(
            "sentence-transformers is required for dense graph retrieval."
        ) from error


def _build_dense_ranker(
    settings: DenseEncoderSettings,
    encoder: SentenceEncoder | None,
    *,
    device: str,
) -> DenseTaskRetriever:
    return DenseTaskRetriever(
        config=DenseConfig(
            model_name=settings.model_name,
            query_prefix=settings.query_prefix,
            passage_prefix=settings.passage_prefix,
            batch_size=settings.batch_size,
            device=device,
        ),
        encoder=encoder or _resolve_encoder(settings, None, device=device),
        device=device,
    )


def _built(
    retrieval_method: RetrievalMethod,
    *,
    method: RetrievalMethodId,
    execution_tasks: list[RetrievalExecutionTask],
    model: Path | None = None,
    device: str | None = None,
    encoder: DenseEncoderSettings | None = None,
) -> BuiltRetrievalMethod:
    return BuiltRetrievalMethod(
        method=retrieval_method,
        provenance=RetrievalProvenance(
            method=method,
            model=model,
            device=device,
            encoder=encoder,
        ),
        execution_tasks=execution_tasks,
    )


def _text_execution_tasks(
    text_requests: list[TextRankingRequest],
) -> list[RetrievalExecutionTask]:
    return [
        RetrievalExecutionTask(text_request=request, method_request=request)
        for request in text_requests
    ]


def _evidence_execution_tasks(
    text_requests: list[TextRankingRequest],
    graph_index: GraphIndex,
    initial_scores_for_request: Callable[[TextRankingRequest], dict[str, float]],
) -> list[RetrievalExecutionTask]:
    tasks: list[RetrievalExecutionTask] = []
    for request in text_requests:
        evidence_request = EvidenceGraphRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
            graph=graph_index.get_required(request.task_id),
            initial_scores=initial_scores_for_request(request),
        )
        tasks.append(
            RetrievalExecutionTask(
                text_request=request,
                method_request=evidence_request,
            )
        )
    return tasks


def _initial_scores_from_seed_signal_provider(
    seed_signal_provider: SeedSignalProvider,
) -> Callable[[TextRankingRequest], dict[str, float]]:
    def initial_scores(request: TextRankingRequest) -> dict[str, float]:
        return {
            signal.node_id: signal.score
            for signal in seed_signal_provider.score_task(request)
        }

    return initial_scores


__all__ = [
    "build_retrieval_registry",
    "seed_retrieval_settings_for_method",
]
