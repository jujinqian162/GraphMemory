from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.graphs.index import GraphIndex
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphBuildPayload,
    CheckpointGraphRetrievalSettings,
    DenseEncoderSettings,
    DenseRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    RETRIEVAL_METHOD_METADATA,
    RetrievalBuilderSpec,
    RetrievalJobSettings,
    RetrievalMethodId,
    RetrievalRegistry,
    SeedRetrieverBuildPayload,
    SeedRetrievalSettings,
    get_retrieval_method_metadata,
    require_payload,
)
from graph_memory.retrieval.contracts import DenseEncoder, RetrievalMethod, SeedRanker
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.validation import validate_graphs, validate_task_id_alignment


class RuntimeRetrievalRegistry(RetrievalRegistry):
    def settings_from_runtime(
        self,
        *,
        method: str,
        top_k: int,
        dense_config: DenseConfig | None = None,
        graph_config: GraphRerankConfig | Mapping[str, object] | None = None,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> RetrievalJobSettings:
        metadata = get_retrieval_method_metadata(method)
        method_id = RetrievalMethodId(metadata.name)
        settings_type = metadata.settings_type

        if settings_type is Bm25RetrievalSettings:
            return Bm25RetrievalSettings(top_k=top_k)
        if settings_type is DenseRetrievalSettings:
            return DenseRetrievalSettings(
                top_k=top_k,
                encoder=_dense_encoder_settings(dense_config),
            )
        if settings_type is GraphRerankRetrievalSettings:
            return GraphRerankRetrievalSettings(
                method=cast(
                    Literal[RetrievalMethodId.BM25_GRAPH_RERANK, RetrievalMethodId.DENSE_GRAPH_RERANK],
                    method_id,
                ),
                top_k=top_k,
                seed=seed_retrieval_settings_for_method(method=method, dense_config=dense_config),
                rerank=_graph_rerank_settings_from_config(graph_config),
            )
        if settings_type is CheckpointGraphRetrievalSettings:
            if checkpoint is None:
                raise ValueError(f"Trainable graph method={method} requires a checkpoint path.")
            return CheckpointGraphRetrievalSettings(top_k=top_k, checkpoint=Path(checkpoint), device=device)
        raise ValueError(f"Unsupported retrieval settings type: {settings_type.__name__}")

    def seed_settings_for_method(
        self,
        method: str,
        dense_config: DenseConfig | None = None,
    ) -> SeedRetrievalSettings:
        return seed_retrieval_settings_for_method(method=method, dense_config=dense_config)


def build_retrieval_registry() -> RuntimeRetrievalRegistry:
    return RuntimeRetrievalRegistry(
        metadata=RETRIEVAL_METHOD_METADATA,
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
            GraphRerankRetrievalSettings: RetrievalBuilderSpec(
                GraphRerankRetrievalSettings,
                lambda settings, deps: _build_graph_rerank(cast(GraphRerankRetrievalSettings, settings), deps),
            ),
            CheckpointGraphRetrievalSettings: RetrievalBuilderSpec(
                CheckpointGraphRetrievalSettings,
                lambda settings, deps: _build_checkpoint_graph(cast(CheckpointGraphRetrievalSettings, settings), deps),
            ),
        },
    )

def seed_retrieval_settings_for_method(
    *,
    method: str,
    dense_config: DenseConfig | None = None,
) -> SeedRetrievalSettings:
    metadata = get_retrieval_method_metadata(method)
    seed_method = metadata.seed_method or RetrievalMethodId(metadata.name)
    if seed_method is RetrievalMethodId.BM25:
        return SeedRetrievalSettings(method=RetrievalMethodId.BM25)
    if seed_method is RetrievalMethodId.DENSE:
        return SeedRetrievalSettings(method=RetrievalMethodId.DENSE, encoder=_dense_encoder_settings(dense_config))
    raise ValueError(f"Unsupported seed retrieval method: {seed_method.value}")


def _dense_encoder_settings(config: DenseConfig | None) -> DenseEncoderSettings:
    if config is None:
        config = DenseConfig()
    return DenseEncoderSettings(
        model_name=config.model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


def _graph_rerank_settings_from_config(value: GraphRerankConfig | Mapping[str, object] | None) -> GraphRerankSettings:
    from graph_memory.retrieval.methods.graph_rerank.config import ensure_graph_rerank_config

    config = ensure_graph_rerank_config(value)
    return GraphRerankSettings(
        lambda_init=config.lambda_init,
        lambda_query=config.lambda_query,
        lambda_neighbor=config.lambda_neighbor,
        lambda_bridge=config.lambda_bridge,
        lambda_path=config.lambda_path,
        seed_top_s=config.seed_top_s,
        max_hops=config.max_hops,
        neighbor_type_weights=dict(config.neighbor_type_weights),
    )


def _build_bm25(settings: Bm25RetrievalSettings, payload: object) -> RetrievalMethod:
    _ = require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    return ScorePipelineMethod(name=settings.method.value, retriever=BM25TaskRetriever())


def _build_dense(settings: DenseRetrievalSettings, payload: object) -> RetrievalMethod:
    build_payload = require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    return ScorePipelineMethod(
        name=settings.method.value,
        retriever=_build_seed_retriever(
            SeedRetrievalSettings(method=RetrievalMethodId.DENSE, encoder=settings.encoder),
            SeedRetrieverBuildPayload(dense_encoder=build_payload.dense_encoder),
        ),
    )


def _build_graph_rerank(settings: GraphRerankRetrievalSettings, payload: object) -> RetrievalMethod:
    from graph_memory.retrieval.methods.graph_rerank.config import ensure_graph_rerank_config
    from graph_memory.retrieval.methods.graph_rerank.method import GraphRerankMethod

    build_payload = require_payload(payload, GraphRerankBuildPayload, method=settings.method.value)
    return GraphRerankMethod(
        name=settings.method.value,
        retriever=_build_seed_retriever(
            settings.seed,
            SeedRetrieverBuildPayload(dense_encoder=build_payload.dense_encoder),
        ),
        graphs=_validated_graph_index(settings.method.value, build_payload.task_inputs, build_payload.graphs),
        graph_config=(
            ensure_graph_rerank_config(cast(GraphRerankConfig | Mapping[str, object] | None, build_payload.graph_config))
            if build_payload.graph_config is not None
            else _graph_rerank_config(settings.rerank)
        ),
    )


def _build_checkpoint_graph(settings: CheckpointGraphRetrievalSettings, payload: object) -> RetrievalMethod:
    from graph_memory.retrieval.methods.trainable_graph import TrainableGraphRetrievalMethod

    build_payload = require_payload(payload, CheckpointGraphBuildPayload, method=settings.method.value)
    text_embedding_provider, seed_signal_provider = _checkpoint_graph_providers(settings, build_payload)
    return TrainableGraphRetrievalMethod.from_checkpoint(
        settings.checkpoint,
        graphs=list(
            _validated_graph_index(
                settings.method.value,
                build_payload.task_inputs,
                build_payload.graphs,
            ).graph_by_task_id.values()
        ),
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        device=settings.device,
    )


def _checkpoint_graph_providers(settings: CheckpointGraphRetrievalSettings, payload: CheckpointGraphBuildPayload):
    if payload.text_embedding_provider is not None and payload.seed_signal_provider is not None:
        return payload.text_embedding_provider, payload.seed_signal_provider

    from graph_memory.models.graph_retriever.checkpoint import load_trainable_checkpoint
    from graph_memory.models.graph_retriever.contracts import SentenceEncoder
    from graph_memory.models.graph_retriever.text_embeddings import DenseTextEmbeddingProvider
    from graph_memory.retrieval.signals import RetrieverSeedSignalProvider

    checkpoint = load_trainable_checkpoint(
        settings.checkpoint,
        expected_method=settings.method.value,
        map_location=settings.device,
    )
    text_embedding_provider = payload.text_embedding_provider
    if text_embedding_provider is None:
        text_embedding_provider = DenseTextEmbeddingProvider(
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
                encoder=cast(DenseEncoder | None, encoder),
            )
        )
    return text_embedding_provider, seed_signal_provider


def _build_seed_retriever(settings: SeedRetrievalSettings, payload: object) -> SeedRanker:
    build_payload = require_payload(payload, SeedRetrieverBuildPayload, method=settings.method.value)
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


def _validated_graph_index(method: str, task_inputs: list[MemoryTaskInput], graphs: list[MemoryGraph]) -> GraphIndex:
    if not graphs:
        raise ValueError(f"Graph-backed retrieval method={method} requires graph inputs.")
    inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
    validate_graphs(graphs, inputs_by_task_id)
    validate_task_id_alignment(
        "retrieval graph inputs",
        set(inputs_by_task_id),
        {graph["task_id"] for graph in graphs},
    )
    return GraphIndex.from_graphs(graphs)


RETRIEVAL_REGISTRY = build_retrieval_registry()


__all__ = [
    "RETRIEVAL_REGISTRY",
    "RuntimeRetrievalRegistry",
    "build_retrieval_registry",
    "seed_retrieval_settings_for_method",
]
