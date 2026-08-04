from __future__ import annotations

import shutil
from typing import cast

from pydantic import JsonValue, TypeAdapter

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.datasets.isetrace.benchmark_records import (
    ISETraceLabelRecord,
    ISETraceQueryMetadata,
    ISETraceRankingRecord,
)
from graph_memory.datasets.isetrace.training import (
    adapt_flat_dense_training_split,
    adapt_provenance_training_split,
)
from graph_memory.datasets.selection import (
    evidence_labels_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    FrozenEmbeddingsArtifactRef,
    ModelArtifactRef,
    ProcessedAssetStore,
    RevisionSourceRef,
    TrainingPairsArtifactRef,
    artifact_payload_path,
)
from graph_memory.experiment.config import (
    DatasetName,
    DenseEncoderConfig,
    DenseFinetuneStageConfig,
    RgcnTrainStageConfig,
)
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.provenance import provenance_rgcn_model_config
from graph_memory.models.graph_retriever.provenance_training import (
    QueryOrigin,
    train_provenance_graph_retriever,
)
from graph_memory.models.graph_retriever.text_embeddings import (
    PrecomputedGraphFeatureProvider,
)
from graph_memory.query_synthesis.provenance.contracts import (
    TemplateSupervisionRecord,
)
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.stages.frozen_embeddings import FrozenEmbeddingStore
from graph_memory.stages.results import ModelResult
from graph_memory.stages.train_payloads import (
    DenseFinetuneTrainPayload,
    RgcnTrainPayload,
)
from graph_memory.stages.trainers import (
    DenseFinetuneMethodTrainer,
    RgcnGraphRetrieverTrainer,
)


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
PROVENANCE_GRAPHS_ADAPTER = TypeAdapter(list[ProvenanceGraph])
ISETRACE_RANKINGS_ADAPTER = TypeAdapter(list[ISETraceRankingRecord])
ISETRACE_LABELS_ADAPTER = TypeAdapter(list[ISETraceLabelRecord])
ISETRACE_QUERY_METADATA_ADAPTER = TypeAdapter(list[ISETraceQueryMetadata])
TEMPLATE_SUPERVISION_ADAPTER = TypeAdapter(list[TemplateSupervisionRecord])
TRAIN_PAIRS_ADAPTER = TypeAdapter(list[TrainPairRecord])


def materialize_dense_finetune_model(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    config: DenseFinetuneStageConfig,
    train_prepared: DatasetArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    encoder_source: EncoderSourceRef,
    implementation_version: str,
) -> ModelResult:
    effective = config.model_copy(
        update={"encoder": _resolved_encoder(config.encoder, encoder_source)}
    )
    train_tasks = cast(
        list[object], read_json(artifact_payload_path(train_prepared, "tasks"))
    )
    train_labels = cast(
        list[object], read_json(artifact_payload_path(train_prepared, "labels"))
    )
    dev_tasks = cast(
        list[object], read_json(artifact_payload_path(dev_prepared, "tasks"))
    )
    dev_labels = cast(
        list[object], read_json(artifact_payload_path(dev_prepared, "labels"))
    )
    pairs = TRAIN_PAIRS_ADAPTER.validate_python(
        read_json(artifact_payload_path(train_pairs, "pairs"))
    )
    train_requests, train_compiled_labels, train_group_ids, _ = _dense_finetune_split(
        dataset,
        train_tasks,
        train_labels,
        metadata=_isetrace_query_metadata(dataset, train_prepared),
    )
    dev_requests, dev_compiled_labels, _, dev_query_origins = _dense_finetune_split(
        dataset,
        dev_tasks,
        dev_labels,
        metadata=_isetrace_query_metadata(dataset, dev_prepared),
    )
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.MODEL,
        namespace=config.method,
        task_identity="train-dense-ft",
        origin={
            "stage": "train",
            "dataset": dataset,
            "method": config.method,
            "prepared_digest": train_prepared.digest,
            "pairs_digest": train_pairs.digest,
            "dev_digest": dev_prepared.digest,
            "encoder_identity": _encoder_identity(encoder_source),
            "implementation_version": implementation_version,
        },
    ) as publisher:
        trainer_output = publisher.workspace / "trainer_output"
        model_dir = publisher.workspace / "model"
        result = DenseFinetuneMethodTrainer(effective).train(
            DenseFinetuneTrainPayload(
                train_requests=tuple(train_requests),
                train_labels=tuple(train_compiled_labels),
                train_pairs=tuple(pairs),
                train_group_ids=train_group_ids,
                dev_requests=tuple(dev_requests),
                dev_labels=tuple(dev_compiled_labels),
                dev_query_origins=dev_query_origins,
                output_dir=trainer_output,
                model_dir=model_dir,
            )
        )
        shutil.rmtree(trainer_output, ignore_errors=True)
        history = tuple(
            cast(dict[str, JsonValue], dict(record)) for record in result.metric_records
        )
        write_jsonl(publisher.workspace / "training_metrics.jsonl", list(history))
        artifact = publisher.publish(
            {
                "model": "model",
                "training_metrics": "training_metrics.jsonl",
            },
            metadata={
                "selected_metric_name": result.selected_metric_name,
                "selected_metric_value": result.selected_metric_value,
                "selection_query_origin": result.selection_query_origin,
                "effective_sampling": train_pairs.origin.get("sampling_config"),
            },
        )
    assert isinstance(artifact, ModelArtifactRef)
    return ModelResult(
        method=config.method,
        artifact=artifact,
        training_history=history,
        metadata={
            "selected_metric_name": result.selected_metric_name,
            "selected_metric_value": result.selected_metric_value,
            "selection_query_origin": result.selection_query_origin,
        },
    )


def _dense_finetune_split(
    dataset: DatasetName,
    tasks: list[object],
    labels: list[object],
    *,
    metadata: list[ISETraceQueryMetadata],
) -> tuple[
    list[TextRankingRequest],
    list[EvidenceLabel],
    dict[str, str],
    dict[str, str],
]:
    if dataset != "isetrace":
        requests = text_ranking_requests_for_dataset(dataset, tasks)
        compiled_labels = evidence_labels_for_dataset(dataset, labels)
        return requests, compiled_labels, {}, {}

    rankings = ISETRACE_RANKINGS_ADAPTER.validate_python(tasks)
    isetrace_labels = ISETRACE_LABELS_ADAPTER.validate_python(labels)
    requests, compiled_labels = adapt_flat_dense_training_split(
        rankings,
        isetrace_labels,
    )
    group_ids = {ranking.task_id: ranking.graph_id for ranking in rankings}
    query_origins = {record.task_id: record.query_origin for record in metadata}
    if set(query_origins) != {request.task_id for request in requests}:
        raise ValueError("ISETrace Dense-FT query origins must align")
    return requests, compiled_labels, group_ids, query_origins


def _isetrace_query_metadata(
    dataset: DatasetName,
    prepared: DatasetArtifactRef,
) -> list[ISETraceQueryMetadata]:
    if dataset != "isetrace":
        return []
    return ISETRACE_QUERY_METADATA_ADAPTER.validate_python(
        read_json(artifact_payload_path(prepared, "query_metadata"))
    )


def materialize_evidence_rgcn_model(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    config: RgcnTrainStageConfig,
    train_prepared: DatasetArtifactRef,
    train_graphs: EvidenceGraphArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    dev_graphs: EvidenceGraphArtifactRef,
    encoder_source: EncoderSourceRef,
    seed_model: ModelArtifactRef | None,
    frozen_embeddings: FrozenEmbeddingsArtifactRef,
    implementation_version: str,
) -> ModelResult:
    method = config.method
    variant = config.variant
    effective_encoder = _resolved_encoder(config.encoder, encoder_source)
    train_tasks = cast(
        list[object], read_json(artifact_payload_path(train_prepared, "tasks"))
    )
    dev_tasks = cast(
        list[object], read_json(artifact_payload_path(dev_prepared, "tasks"))
    )
    train_labels = cast(
        list[object], read_json(artifact_payload_path(train_prepared, "labels"))
    )
    dev_labels = cast(
        list[object], read_json(artifact_payload_path(dev_prepared, "labels"))
    )
    train_graph_values = EVIDENCE_GRAPHS_ADAPTER.validate_python(
        read_json(artifact_payload_path(train_graphs, "graphs"))
    )
    dev_graph_values = EVIDENCE_GRAPHS_ADAPTER.validate_python(
        read_json(artifact_payload_path(dev_graphs, "graphs"))
    )
    pair_values = TRAIN_PAIRS_ADAPTER.validate_python(
        read_json(artifact_payload_path(train_pairs, "pairs"))
    )
    seed_dir = (
        artifact_payload_path(seed_model, "model")
        if seed_model is not None
        else None
    )
    embedding_store = FrozenEmbeddingStore(frozen_embeddings)
    if embedding_store.index.family != "evidence":
        raise ValueError("Evidence R-GCN requires evidence frozen embeddings.")
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.MODEL,
        namespace=method,
        task_identity=f"train-{method}",
        origin={
            "stage": "train",
            "dataset": dataset,
            "method": method,
            "variant": variant,
            "prepared_digest": train_prepared.digest,
            "graph_digest": train_graphs.digest,
            "pairs_digest": train_pairs.digest,
            "dev_digest": dev_prepared.digest,
            "dev_graph_digest": dev_graphs.digest,
            "encoder_identity": _encoder_identity(encoder_source),
            "seed_model_digest": None if seed_model is None else seed_model.digest,
            "frozen_embeddings_digest": frozen_embeddings.digest,
            "implementation_version": implementation_version,
        },
    ) as publisher:
        result = RgcnGraphRetrieverTrainer(
            method=method,
            encoder=effective_encoder,
            train_config=config.train,
            seed_checkpoint=seed_dir,
            train_embeddings=embedding_store.partition("train"),
            dev_embeddings=embedding_store.partition("dev"),
        ).train(
            RgcnTrainPayload(
                train_requests=tuple(
                    text_ranking_requests_for_dataset(dataset, train_tasks)
                ),
                train_labels=tuple(
                    evidence_labels_for_dataset(dataset, train_labels)
                ),
                train_graphs=tuple(train_graph_values),
                train_pairs=tuple(pair_values),
                dev_requests=tuple(
                    text_ranking_requests_for_dataset(dataset, dev_tasks)
                ),
                dev_labels=tuple(
                    evidence_labels_for_dataset(dataset, dev_labels)
                ),
                dev_graphs=tuple(dev_graph_values),
            )
        )
        checkpoints = publisher.workspace / "checkpoints"
        best_model = build_model_from_config(result.model_config)
        best_model.load_state_dict(result.best_model_state_dict)
        epoch_checkpoint = checkpoints / f"checkpoint_epoch_{result.best_epoch}.pt"
        for path in (epoch_checkpoint, checkpoints / "best.pt"):
            save_rgcn_checkpoint(
                path,
                method_name=result.model_config.method_name,
                model=best_model,
                optimizer_state_dict=result.optimizer_state_dict,
                scheduler_state_dict=result.scheduler_state_dict,
                epoch=result.best_epoch,
                global_step=result.global_step,
                best_dev_metric=result.best_dev_metric,
                model_config=result.model_config,
                training_config=result.training_config,
            )
        history = tuple(
            cast(dict[str, JsonValue], dict(record)) for record in result.metric_records
        )
        write_jsonl(publisher.workspace / "training_metrics.jsonl", list(history))
        artifact = publisher.publish(
            {
                "checkpoints": "checkpoints",
                "checkpoint": "checkpoints/best.pt",
                "training_metrics": "training_metrics.jsonl",
            },
            metadata={
                "variant": variant,
                "best_epoch": result.best_epoch,
                "global_step": result.global_step,
                "best_dev_metric": result.best_dev_metric,
            },
        )
    assert isinstance(artifact, ModelArtifactRef)
    return ModelResult(
        method=method,
        artifact=artifact,
        training_history=history,
        metadata={
            "variant": variant,
            "best_epoch": result.best_epoch,
            "global_step": result.global_step,
            "best_dev_metric": result.best_dev_metric,
        },
    )



def materialize_provenance_rgcn_model(
    store: ProcessedAssetStore,
    *,
    config: RgcnTrainStageConfig,
    train_prepared: DatasetArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    encoder_source: EncoderSourceRef,
    frozen_embeddings: FrozenEmbeddingsArtifactRef,
    implementation_version: str,
) -> ModelResult:
    if config.method != "provenance_rgcn":
        raise ValueError("provenance model stage requires method=provenance_rgcn")
    effective_encoder = _resolved_encoder(config.encoder, encoder_source)
    train_requests, train_labels, _train_origins = _load_provenance_split(
        train_prepared
    )
    dev_requests, dev_labels, dev_origins = _load_provenance_split(dev_prepared)
    selection_query_origin = (
        "natural" if "natural" in dev_origins.values() else "template"
    )
    selection_metric = f"dev_{selection_query_origin}_recall_at_5"
    pair_values = TRAIN_PAIRS_ADAPTER.validate_python(
        read_json(artifact_payload_path(train_pairs, "pairs"))
    )
    embedding_store = FrozenEmbeddingStore(frozen_embeddings)
    if embedding_store.index.family != "provenance":
        raise ValueError("Provenance R-GCN requires provenance frozen embeddings.")
    train_partition = embedding_store.partition("train")
    dev_partition = embedding_store.partition("dev")
    encoder_dim = embedding_store.index.embedding_dim
    train_provider = PrecomputedGraphFeatureProvider(
        train_partition, embedding_dim=encoder_dim
    )
    dev_provider = PrecomputedGraphFeatureProvider(
        dev_partition, embedding_dim=encoder_dim
    )
    model_settings = config.train.model
    model_config = provenance_rgcn_model_config(
        encoder_model=effective_encoder.model_name,
        encoder_dim=encoder_dim,
        query_prefix=effective_encoder.query_prefix,
        passage_prefix=effective_encoder.passage_prefix,
        encoder_batch_size=effective_encoder.batch_size,
        hidden_dim=model_settings.hidden_dim,
        num_layers=model_settings.num_layers,
        dropout=model_settings.dropout,
        ablation_name=model_settings.ablation,
    )
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.MODEL,
        namespace=config.method,
        task_identity="train-provenance-rgcn",
        origin={
            "stage": "train",
            "dataset": "isetrace",
            "method": config.method,
            "variant": config.variant,
            "prepared_digest": train_prepared.digest,
            "pairs_digest": train_pairs.digest,
            "dev_digest": dev_prepared.digest,
            "encoder_identity": _encoder_identity(encoder_source),
            "frozen_embeddings_digest": frozen_embeddings.digest,
            "implementation_version": implementation_version,
        },
    ) as publisher:
        result = train_provenance_graph_retriever(
            train_requests=train_requests,
            train_labels=train_labels,
            train_pairs=pair_values,
            dev_requests=dev_requests,
            dev_labels=dev_labels,
            dev_query_origins=dev_origins,
            model_config=model_config,
            training_config=config.train.trainer,
            text_embedding_provider=train_provider,
            dev_text_embedding_provider=dev_provider,
            device=config.train.trainer.device,
        )
        checkpoints = publisher.workspace / "checkpoints"
        best_model = build_model_from_config(result.model_config)
        best_model.load_state_dict(result.best_model_state_dict)
        epoch_checkpoint = checkpoints / f"checkpoint_epoch_{result.best_epoch}.pt"
        for path in (epoch_checkpoint, checkpoints / "best.pt"):
            save_rgcn_checkpoint(
                path,
                method_name=result.model_config.method_name,
                model=best_model,
                optimizer_state_dict=result.optimizer_state_dict,
                scheduler_state_dict=result.scheduler_state_dict,
                epoch=result.best_epoch,
                global_step=result.global_step,
                best_dev_metric=result.best_dev_metric,
                model_config=result.model_config,
                training_config=result.training_config,
            )
        history = tuple(
            cast(dict[str, JsonValue], dict(record)) for record in result.metric_records
        )
        write_jsonl(publisher.workspace / "training_metrics.jsonl", list(history))
        artifact = publisher.publish(
            {
                "checkpoints": "checkpoints",
                "checkpoint": "checkpoints/best.pt",
                "training_metrics": "training_metrics.jsonl",
            },
            metadata={
                "variant": config.variant,
                "best_epoch": result.best_epoch,
                "global_step": result.global_step,
                "best_dev_metric": result.best_dev_metric,
                "selection_query_origin": selection_query_origin,
                "selection_metric": selection_metric,
            },
        )
    assert isinstance(artifact, ModelArtifactRef)
    metadata: dict[str, JsonValue] = {
        "variant": config.variant,
        "best_epoch": result.best_epoch,
        "global_step": result.global_step,
        "best_dev_metric": result.best_dev_metric,
        "selection_query_origin": selection_query_origin,
        "selection_metric": selection_metric,
    }
    return ModelResult(
        method=config.method,
        artifact=artifact,
        training_history=history,
        metadata=metadata,
    )


def _load_provenance_split(prepared: DatasetArtifactRef):
    rankings = ISETRACE_RANKINGS_ADAPTER.validate_python(
        read_json(artifact_payload_path(prepared, "tasks"))
    )
    labels = ISETRACE_LABELS_ADAPTER.validate_python(
        read_json(artifact_payload_path(prepared, "labels"))
    )
    graphs = PROVENANCE_GRAPHS_ADAPTER.validate_python(
        read_json(artifact_payload_path(prepared, "provenance_graphs"))
    )
    templates = TEMPLATE_SUPERVISION_ADAPTER.validate_python(
        read_json(artifact_payload_path(prepared, "template_supervision"))
    )
    metadata = ISETRACE_QUERY_METADATA_ADAPTER.validate_python(
        read_json(artifact_payload_path(prepared, "query_metadata"))
    )
    requests, compiled_labels = adapt_provenance_training_split(
        rankings, labels, graphs, templates
    )
    origins = cast(
        dict[str, QueryOrigin],
        {record.task_id: record.query_origin for record in metadata},
    )
    if set(origins) != {request.task_id for request in requests}:
        raise ValueError("ISETrace provenance query origins must align")
    return requests, compiled_labels, origins


def _resolved_encoder(
    encoder: DenseEncoderConfig,
    source: EncoderSourceRef,
) -> DenseEncoderConfig:
    if isinstance(source, RevisionSourceRef):
        return encoder
    return encoder.model_copy(update={"model_name": source.uri})


def _encoder_identity(source: EncoderSourceRef) -> str:
    if isinstance(source, (FileSourceRef, DirectorySourceRef)):
        return source.digest
    return source.revision


__all__ = [
    "EncoderSourceRef",
    "materialize_dense_finetune_model",
    "materialize_evidence_rgcn_model",
    "materialize_provenance_rgcn_model",
]
