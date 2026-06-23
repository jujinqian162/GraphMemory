from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.embeddings import SentenceEncoder, load_sentence_transformer
from graph_memory.graphs.index import GraphIndex
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    BuiltRetrievalMethod,
    CheckpointGraphBuildPayload,
    CheckpointGraphRetrievalSettings,
    DenseEncoderSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    FastGraphRAGBuildPayload,
    FastGraphRAGRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    ImportanceArtifactProvenance,
    MemoryStreamBuildPayload,
    MemoryStreamRetrievalSettings,
    RetrievalBuilderSpec,
    RetrievalMethodId,
    RetrievalProvenance,
    RetrievalRegistry,
    SeedRetrieverBuildPayload,
    SeedRetrievalSettings,
    _require_payload,
)
from graph_memory.retrieval.contracts import RetrievalMethod, SeedRanker
from graph_memory.retrieval.execution.requests import RetrievalExecutionTask
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig
from graph_memory.retrieval.requests import (
    DenseConfigLike,
    FastGraphRAGKnowledgeGraph,
    FastGraphRAGRequest,
    GraphRankingRequest,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
from graph_memory.validation import validate_graphs, validate_task_id_alignment
from graph_memory.validation import select_importance_records
from graph_memory.models.dense_finetune.metadata import load_dense_ft_model_metadata


def build_retrieval_registry() -> RetrievalRegistry:
    return RetrievalRegistry(
        seed_build=_build_seed_retriever,
        builders={
            Bm25RetrievalSettings: RetrievalBuilderSpec(
                Bm25RetrievalSettings,
                lambda settings, deps: _build_bm25(cast(Bm25RetrievalSettings, settings), deps),
            ),
            DenseRetrievalSettings: RetrievalBuilderSpec(
                DenseRetrievalSettings,
                lambda settings, deps: _build_dense(cast(DenseRetrievalSettings, settings), deps),
            ),
            MemoryStreamRetrievalSettings: RetrievalBuilderSpec(
                MemoryStreamRetrievalSettings,
                lambda settings, deps: _build_memory_stream(cast(MemoryStreamRetrievalSettings, settings), deps),
            ),
            FastGraphRAGRetrievalSettings: RetrievalBuilderSpec(
                FastGraphRAGRetrievalSettings,
                lambda settings, deps: _build_fast_graphrag(cast(FastGraphRAGRetrievalSettings, settings), deps),
            ),
            GraphRerankRetrievalSettings: RetrievalBuilderSpec(
                GraphRerankRetrievalSettings,
                lambda settings, deps: _build_graph_rerank(cast(GraphRerankRetrievalSettings, settings), deps),
            ),
            CheckpointGraphRetrievalSettings: RetrievalBuilderSpec(
                CheckpointGraphRetrievalSettings,
                lambda settings, deps: _build_checkpoint_graph(cast(CheckpointGraphRetrievalSettings, settings), deps),
            ),
            DenseFinetunedRetrievalSettings: RetrievalBuilderSpec(
                DenseFinetunedRetrievalSettings,
                lambda settings, deps: _build_dense_ft(cast(DenseFinetunedRetrievalSettings, settings), deps),
            ),
        },
    )


def seed_retrieval_settings_for_method(
    *,
    method: RetrievalMethodId,
    dense_config: DenseConfigLike | None = None,
) -> SeedRetrievalSettings:
    if method is RetrievalMethodId.BM25:
        return SeedRetrievalSettings(method=RetrievalMethodId.BM25)
    if method is RetrievalMethodId.DENSE:
        return SeedRetrievalSettings(method=RetrievalMethodId.DENSE, encoder=_dense_encoder_settings(dense_config))
    raise ValueError(f"Unsupported seed retrieval method: {method.value}")


def _dense_encoder_settings(config: DenseConfigLike | None) -> DenseEncoderSettings:
    if config is None:
        config = DenseConfig()
    return DenseEncoderSettings(
        model_name=config.model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


def _build_bm25(settings: Bm25RetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    build_payload = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    return _built(
        ScorePipelineMethod(name=settings.method.value, retriever=BM25TaskRetriever()),
        method=settings.method,
        execution_tasks=_text_execution_tasks(build_payload.ranking_requests),
    )


def _build_dense(settings: DenseRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    build_payload = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    return _built(
        ScorePipelineMethod(
            name=settings.method.value,
            retriever=_build_seed_retriever(
                SeedRetrievalSettings(method=RetrievalMethodId.DENSE, encoder=settings.encoder),
                SeedRetrieverBuildPayload(dense_encoder=build_payload.dense_encoder),
            ),
        ),
        method=settings.method,
        encoder=settings.encoder,
        execution_tasks=_text_execution_tasks(build_payload.ranking_requests),
    )


def _build_memory_stream(settings: MemoryStreamRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    build_payload = _require_payload(payload, MemoryStreamBuildPayload, method=settings.method.value)
    importance_by_task_id = _select_importance_records_for_memory_stream(settings, build_payload)
    dense_seed_ranker = _build_seed_retriever(
        SeedRetrievalSettings(method=RetrievalMethodId.DENSE, encoder=settings.encoder),
        SeedRetrieverBuildPayload(dense_encoder=build_payload.dense_encoder),
    )
    return _built(
        MemoryStreamMethod(
            name=settings.method.value,
            dense_seed_ranker=dense_seed_ranker,
            importance_by_task_id=importance_by_task_id,
            scoring=build_payload.scoring_config or settings.scoring,
        ),
        method=settings.method,
        encoder=settings.encoder,
        importance=ImportanceArtifactProvenance(
            path=build_payload.importance_path,
            sha256=build_payload.importance_sha256,
            schema_version=1,
        ),
        execution_tasks=_memory_stream_execution_tasks(build_payload.temporal_requests, importance_by_task_id),
    )


def _build_fast_graphrag(settings: FastGraphRAGRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.fast_graphrag.index import build_fast_graphrag_knowledge_graph
    from graph_memory.retrieval.methods.fast_graphrag.method import (
        DenseFastGraphRAGScorer,
        FastGraphRAGMethod,
    )

    build_payload = _require_payload(payload, FastGraphRAGBuildPayload, method=settings.method.value)
    graph_index = _validated_graph_index(settings.method.value, build_payload.ranking_requests, build_payload.graphs)
    method_config = FastGraphRAGConfig(
        extraction=settings.extraction,
        pruning=settings.pruning,
        scoring=settings.scoring,
        entity_seed_top_k=settings.entity_seed_top_k,
        query_link_seed_score=settings.query_link_seed_score,
        dense_entity_seed_weight=settings.dense_entity_seed_weight,
        lexical_substring_match_score=settings.lexical_substring_match_score,
        ppr_damping=settings.ppr_damping,
        ppr_max_iterations=settings.ppr_max_iterations,
        ppr_tolerance=settings.ppr_tolerance,
    )
    method = FastGraphRAGMethod(
        name=settings.method.value,
        config=method_config,
        dense_ranker=DenseFastGraphRAGScorer(
            config=settings.encoder,
            encoder=build_payload.dense_encoder,
        ),
    )
    return _built(
        method,
        method=settings.method,
        encoder=settings.encoder,
        execution_tasks=_fast_graphrag_execution_tasks(
            build_payload.ranking_requests,
            graph_index,
            build_fast_graphrag_knowledge_graph,
            method_config,
        ),
    )


def _select_importance_records_for_memory_stream(
    settings: MemoryStreamRetrievalSettings,
    payload: MemoryStreamBuildPayload,
) -> Mapping[str, TaskImportanceRecord]:
    """Select and validate current-task importance before method construction."""
    _ = settings
    selected_records = select_importance_records(payload.importance_artifact, payload.temporal_requests)
    return {record["task_id"]: record for record in selected_records}


def _build_graph_rerank(settings: GraphRerankRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.graph_rerank.config import ensure_graph_rerank_config
    from graph_memory.retrieval.methods.graph_rerank.method import GraphRerankMethod

    build_payload = _require_payload(payload, GraphRerankBuildPayload, method=settings.method.value)
    seed_ranker = _build_seed_retriever(
        settings.seed,
        SeedRetrieverBuildPayload(dense_encoder=build_payload.dense_encoder),
    )
    graph_index = _validated_graph_index(settings.method.value, build_payload.ranking_requests, build_payload.graphs)
    graph_config = (
        ensure_graph_rerank_config(cast(GraphRerankConfig | Mapping[str, object] | None, build_payload.graph_config))
        if build_payload.graph_config is not None
        else _graph_rerank_config(settings.rerank)
    )
    return _built(
        GraphRerankMethod(
            name=settings.method.value,
            retriever=seed_ranker,
            graphs=graph_index,
            graph_config=graph_config,
        ),
        method=settings.method,
        encoder=settings.seed.encoder,
        execution_tasks=_graph_execution_tasks(
            build_payload.ranking_requests,
            graph_index,
            _initial_scores_from_seed_ranker(seed_ranker),
        ),
    )


def _build_checkpoint_graph(settings: CheckpointGraphRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.trainable_graph import TrainableGraphRetrievalMethod

    build_payload = _require_payload(payload, CheckpointGraphBuildPayload, method=settings.method.value)
    graph_index = _validated_graph_index(settings.method.value, build_payload.ranking_requests, build_payload.graphs)
    text_embedding_provider, seed_signal_provider, checkpoint = _checkpoint_graph_providers(settings, build_payload)
    method = TrainableGraphRetrievalMethod.from_checkpoint(
        settings.checkpoint,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=settings.device,
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
        execution_tasks=_graph_execution_tasks(
            build_payload.ranking_requests,
            graph_index,
            _initial_scores_from_seed_signal_provider(seed_signal_provider),
        ),
    )


def _build_dense_ft(settings: DenseFinetunedRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    build_payload = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    metadata = load_dense_ft_model_metadata(settings.checkpoint)
    encoder = build_payload.dense_encoder
    if encoder is None:
        try:
            encoder = cast(
                SentenceEncoder,
                cast(object, load_sentence_transformer(settings.checkpoint, device=settings.device)),
            )
        except RuntimeError as error:
            raise RuntimeError("sentence-transformers is required for dense-ft retrieval.") from error
    method = ScorePipelineMethod(
        name=settings.method.value,
        retriever=DenseTaskRetriever(
            config=DenseConfig(
                model_name=str(settings.checkpoint),
                query_prefix=metadata.query_prefix,
                passage_prefix=metadata.passage_prefix,
                batch_size=metadata.batch_size,
            ),
            encoder=encoder,
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
        execution_tasks=_text_execution_tasks(build_payload.ranking_requests),
    )


def _checkpoint_graph_providers(settings: CheckpointGraphRetrievalSettings, payload: CheckpointGraphBuildPayload):
    from graph_memory.models.graph_retriever.checkpoint import load_rgcn_checkpoint
    from graph_memory.models.graph_retriever.text_embeddings import DenseGraphFeatureProvider
    from graph_memory.retrieval.signals import RetrieverSeedSignalProvider

    checkpoint = load_rgcn_checkpoint(
        settings.checkpoint,
        expected_method=settings.method.value,
        map_location="cpu",
    )
    if payload.text_embedding_provider is not None and payload.seed_signal_provider is not None:
        return payload.text_embedding_provider, payload.seed_signal_provider, checkpoint
    if payload.text_embedding_provider is None and payload.seed_signal_provider is None:
        joint_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            encoder=cast(SentenceEncoder | None, payload.dense_encoder),
        )
        return joint_provider, joint_provider, checkpoint

    text_embedding_provider = payload.text_embedding_provider
    if text_embedding_provider is None:
        text_embedding_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            encoder=cast(SentenceEncoder | None, payload.dense_encoder),
        )

    seed_signal_provider = payload.seed_signal_provider
    if seed_signal_provider is None:
        encoder = getattr(text_embedding_provider, "encoder", payload.dense_encoder)
        seed_signal_provider = RetrieverSeedSignalProvider(
            DenseTaskRetriever(
                model_name=checkpoint.model_config.encoder_model,
                query_prefix=checkpoint.model_config.query_prefix,
                passage_prefix=checkpoint.model_config.passage_prefix,
                encoder=cast(SentenceEncoder | None, encoder),
            )
        )
    return text_embedding_provider, seed_signal_provider, checkpoint


def _built(
    retrieval_method: RetrievalMethod,
    *,
    method: RetrievalMethodId,
    execution_tasks: list[RetrievalExecutionTask],
    model: Path | None = None,
    device: str | None = None,
    encoder: DenseEncoderSettings | None = None,
    importance: ImportanceArtifactProvenance | None = None,
) -> BuiltRetrievalMethod:
    return BuiltRetrievalMethod(
        method=retrieval_method,
        provenance=RetrievalProvenance(
            method=method,
            model=model,
            device=device,
            encoder=encoder,
            importance=importance,
        ),
        execution_tasks=execution_tasks,
    )


def _build_seed_retriever(settings: SeedRetrievalSettings, payload: object) -> SeedRanker:
    build_payload = _require_payload(payload, SeedRetrieverBuildPayload, method=settings.method.value)
    if settings.method is RetrievalMethodId.BM25:
        return BM25TaskRetriever()
    if settings.encoder is None:
        raise ValueError("Dense seed retrieval requires encoder settings.")
    return DenseTaskRetriever(
        config=DenseConfig(
            model_name=settings.encoder.model_name,
            query_prefix=settings.encoder.query_prefix,
            passage_prefix=settings.encoder.passage_prefix,
            batch_size=settings.encoder.batch_size,
        ),
        encoder=build_payload.dense_encoder,
    )


def _graph_rerank_config(settings: GraphRerankSettings) -> GraphRerankConfig:
    return GraphRerankConfig(
        lambda_init=settings.lambda_init,
        lambda_query=settings.lambda_query,
        lambda_neighbor=settings.lambda_neighbor,
        lambda_bridge=settings.lambda_bridge,
        lambda_path=settings.lambda_path,
        seed_top_s=settings.seed_top_s,
        max_hops=settings.max_hops,
        neighbor_type_weights=dict(settings.neighbor_type_weights),
    )


def _validated_graph_index(method: str, ranking_requests: list[TextRankingRequest], graphs: list[MemoryGraph]) -> GraphIndex:
    if not graphs:
        raise ValueError(f"Graph-backed retrieval method={method} requires graph inputs.")
    requests_by_task_id = {request.task_id: request for request in ranking_requests}
    validate_graphs(graphs, ranking_requests)
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(requests_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    return GraphIndex.from_graphs(graphs)

def _text_execution_tasks(ranking_requests: list[TextRankingRequest]) -> list[RetrievalExecutionTask]:
    return [
        RetrievalExecutionTask(text_request=request, method_request=request)
        for request in ranking_requests
    ]


def _graph_execution_tasks(
    ranking_requests: list[TextRankingRequest],
    graph_index: GraphIndex,
    initial_scores_for_request: Callable[[TextRankingRequest], dict[str, float]],
) -> list[RetrievalExecutionTask]:
    tasks: list[RetrievalExecutionTask] = []
    for request in ranking_requests:
        graph_request = GraphRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
            graph=graph_index.get_required(request.task_id),
            initial_scores=initial_scores_for_request(request),
        )
        tasks.append(RetrievalExecutionTask(text_request=request, method_request=graph_request))
    return tasks


def _fast_graphrag_execution_tasks(
    ranking_requests: list[TextRankingRequest],
    graph_index: GraphIndex,
    knowledge_graph_builder: Callable[..., FastGraphRAGKnowledgeGraph],
    config: FastGraphRAGConfig,
) -> list[RetrievalExecutionTask]:
    tasks: list[RetrievalExecutionTask] = []
    for request in ranking_requests:
        graph = graph_index.get_required(request.task_id)
        method_request = FastGraphRAGRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
            candidate_graph=graph,
            knowledge_graph=knowledge_graph_builder(request, graph, config=config),
        )
        tasks.append(RetrievalExecutionTask(text_request=request, method_request=method_request))
    return tasks


def _initial_scores_from_seed_ranker(seed_ranker: SeedRanker) -> Callable[[TextRankingRequest], dict[str, float]]:
    def initial_scores(request: TextRankingRequest) -> dict[str, float]:
        return {
            ranked_node.node_id: ranked_node.score
            for ranked_node in seed_ranker.rank(request)
        }

    return initial_scores


def _initial_scores_from_seed_signal_provider(seed_signal_provider: "SeedSignalProvider") -> Callable[[TextRankingRequest], dict[str, float]]:
    def initial_scores(request: TextRankingRequest) -> dict[str, float]:
        return {
            signal.node_id: signal.score
            for signal in seed_signal_provider.score_task(request)
        }

    return initial_scores


def _memory_stream_execution_tasks(
    temporal_requests: list[TemporalMemoryRankingRequest],
    importance_by_task_id: Mapping[str, TaskImportanceRecord],
) -> list[RetrievalExecutionTask]:
    tasks: list[RetrievalExecutionTask] = []
    for request in temporal_requests:
        try:
            task_importance = importance_by_task_id[request.task_id]
        except KeyError as error:
            raise ValueError(f"Missing importance record for task_id={request.task_id}.") from error
        method_request = TemporalMemoryRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
            importance_by_item_id={node_id: float(score) for node_id, score in task_importance["scores"].items()},
            metadata=request.metadata,
        )
        text_request = TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
        )
        tasks.append(RetrievalExecutionTask(text_request=text_request, method_request=method_request))
    return tasks

__all__ = [
    "build_retrieval_registry",
    "seed_retrieval_settings_for_method",
]
