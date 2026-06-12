from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.embeddings import SentenceEncoder, load_sentence_transformer
from graph_memory.graphs.index import GraphIndex
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    CheckpointGraphBuildPayload,
    CheckpointGraphRetrievalSettings,
    DenseEncoderSettings,
    DenseFinetunedRetrievalSettings,
    DenseRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    RETRIEVAL_METHOD_METADATA,
    RetrievalBuilderSpec,
    RetrievalMethodId,
    RetrievalRegistry,
    SeedRetrieverBuildPayload,
    SeedRetrievalSettings,
    get_retrieval_method_metadata,
    _require_payload,
)
from graph_memory.retrieval.contracts import RetrievalMethod, SeedRanker
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.methods.flat.method import ScorePipelineMethod
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.validation import validate_graphs, validate_task_id_alignment
from graph_memory.models.dense_finetune.training import DENSE_FT_METADATA_FILENAME


def build_retrieval_registry() -> RetrievalRegistry:
    return RetrievalRegistry(
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
            DenseFinetunedRetrievalSettings: RetrievalBuilderSpec(
                DenseFinetunedRetrievalSettings,
                lambda settings, deps: _build_dense_ft(cast(DenseFinetunedRetrievalSettings, settings), deps),
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


def _build_bm25(settings: Bm25RetrievalSettings, payload: object) -> RetrievalMethod:
    _ = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    return ScorePipelineMethod(name=settings.method.value, retriever=BM25TaskRetriever())


def _build_dense(settings: DenseRetrievalSettings, payload: object) -> RetrievalMethod:
    build_payload = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
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

    build_payload = _require_payload(payload, GraphRerankBuildPayload, method=settings.method.value)
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

    build_payload = _require_payload(payload, CheckpointGraphBuildPayload, method=settings.method.value)
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


def _build_dense_ft(settings: DenseFinetunedRetrievalSettings, payload: object) -> RetrievalMethod:
    build_payload = _require_payload(payload, FlatRetrievalBuildPayload, method=settings.method.value)
    metadata = _load_dense_ft_metadata(settings.checkpoint)
    encoder = build_payload.dense_encoder
    if encoder is None:
        try:
            encoder = cast(
                SentenceEncoder,
                cast(object, load_sentence_transformer(settings.checkpoint, device=settings.device)),
            )
        except RuntimeError as error:
            raise RuntimeError("sentence-transformers is required for dense-ft retrieval.") from error
    return ScorePipelineMethod(
        name=settings.method.value,
        retriever=DenseTaskRetriever(
            config=DenseConfig(
                model_name=str(settings.checkpoint),
                query_prefix=_metadata_string(metadata, "query_prefix", settings.checkpoint),
                passage_prefix=_metadata_string(metadata, "passage_prefix", settings.checkpoint),
                batch_size=_metadata_int(metadata, "batch_size", settings.checkpoint),
            ),
            encoder=encoder,
        ),
    )


def _load_dense_ft_metadata(checkpoint: Path) -> dict[str, object]:
    metadata_path = checkpoint / DENSE_FT_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError(f"Missing {DENSE_FT_METADATA_FILENAME} for dense_ft checkpoint: {checkpoint}")
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {DENSE_FT_METADATA_FILENAME} for dense_ft checkpoint: {checkpoint}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{DENSE_FT_METADATA_FILENAME} must contain an object: {checkpoint}")
    method = data.get("method")
    if method != RetrievalMethodId.DENSE_FT.value:
        raise ValueError(f"{DENSE_FT_METADATA_FILENAME} method must be dense_ft for checkpoint: {checkpoint}")
    return data


def _metadata_string(metadata: Mapping[str, object], key: str, checkpoint: Path) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{DENSE_FT_METADATA_FILENAME} field must be a string: {key} ({checkpoint})")
    return value


def _metadata_int(metadata: Mapping[str, object], key: str, checkpoint: Path) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{DENSE_FT_METADATA_FILENAME} field must be an integer: {key} ({checkpoint})")
    return value


def _checkpoint_graph_providers(settings: CheckpointGraphRetrievalSettings, payload: CheckpointGraphBuildPayload):
    if payload.text_embedding_provider is not None and payload.seed_signal_provider is not None:
        return payload.text_embedding_provider, payload.seed_signal_provider

    from graph_memory.models.graph_retriever.checkpoint import load_trainable_checkpoint
    from graph_memory.models.graph_retriever.text_embeddings import DenseGraphFeatureProvider
    from graph_memory.retrieval.signals import RetrieverSeedSignalProvider

    checkpoint = load_trainable_checkpoint(
        settings.checkpoint,
        expected_method=settings.method.value,
        map_location=settings.device,
    )
    if payload.text_embedding_provider is None and payload.seed_signal_provider is None:
        joint_provider = DenseGraphFeatureProvider(
            model_name=checkpoint.model_config.encoder_model,
            query_prefix=checkpoint.model_config.query_prefix,
            passage_prefix=checkpoint.model_config.passage_prefix,
            encoder=cast(SentenceEncoder | None, payload.dense_encoder),
        )
        return joint_provider, joint_provider

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
    return text_embedding_provider, seed_signal_provider


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
