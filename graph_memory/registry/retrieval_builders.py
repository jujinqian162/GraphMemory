from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
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
    dense_config: DenseConfig | None = None,
) -> SeedRetrievalSettings:
    if method is RetrievalMethodId.BM25:
        return SeedRetrievalSettings(method=RetrievalMethodId.BM25)
    if method is RetrievalMethodId.DENSE:
        return SeedRetrievalSettings(method=RetrievalMethodId.DENSE, encoder=_dense_encoder_settings(dense_config))
    raise ValueError(f"Unsupported seed retrieval method: {method.value}")


def _dense_encoder_settings(config: DenseConfig | None) -> DenseEncoderSettings:
    if config is None:
        config = DenseConfig()
    return DenseEncoderSettings(
        model_name=config.model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


def _build_bm25(settings: Bm25RetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    _ = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    return _built(
        ScorePipelineMethod(name=settings.method.value, retriever=BM25TaskRetriever()),
        method=settings.method,
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
            scoring=settings.scoring,
        ),
        method=settings.method,
        encoder=settings.encoder,
        importance=ImportanceArtifactProvenance(
            path=build_payload.importance_path,
            sha256=build_payload.importance_sha256,
            schema_version=1,
        ),
    )


def _select_importance_records_for_memory_stream(
    settings: MemoryStreamRetrievalSettings,
    payload: MemoryStreamBuildPayload,
) -> Mapping[str, TaskImportanceRecord]:
    """Select and validate current-task importance before method construction."""
    _ = settings
    selected_records = select_importance_records(payload.importance_artifact, payload.task_inputs)
    return {record["task_id"]: record for record in selected_records}


def _build_graph_rerank(settings: GraphRerankRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.graph_rerank.config import ensure_graph_rerank_config
    from graph_memory.retrieval.methods.graph_rerank.method import GraphRerankMethod

    build_payload = _require_payload(payload, GraphRerankBuildPayload, method=settings.method.value)
    return _built(
        GraphRerankMethod(
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
        ),
        method=settings.method,
        encoder=settings.seed.encoder,
    )


def _build_checkpoint_graph(settings: CheckpointGraphRetrievalSettings, payload: object) -> BuiltRetrievalMethod:
    from graph_memory.retrieval.methods.trainable_graph import TrainableGraphRetrievalMethod

    build_payload = _require_payload(payload, CheckpointGraphBuildPayload, method=settings.method.value)
    text_embedding_provider, seed_signal_provider, checkpoint = _checkpoint_graph_providers(settings, build_payload)
    method = TrainableGraphRetrievalMethod.from_checkpoint(
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


__all__ = [
    "build_retrieval_registry",
    "seed_retrieval_settings_for_method",
]
