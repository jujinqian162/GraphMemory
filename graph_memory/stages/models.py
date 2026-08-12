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
    ISETraceRankingRecord,
)
from graph_memory.datasets.isetrace.training import (
    adapt_flat_dense_training_split,
    adapt_provenance_training_split,
    adapt_provenance_unit_dense_training_split,
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
    immutable_source_identity,
)
from graph_memory.experiment.config import (
    DatasetName,
    DenseCandidateView,
    DenseEncoderConfig,
    DenseFinetuneMethodConfig,
    ProvenanceRgcnMethodConfig,
    RgcnStageConfig,
)
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.dense_finetune.metadata import load_dense_ft_model_metadata
from graph_memory.models.dense_finetune.training import (
    DenseFinetuneRunConfig,
    train_dense_finetune,
)
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.config.defaults import default_model_config
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.provenance import provenance_rgcn_model_config
from graph_memory.models.graph_retriever.provenance_training import (
    train_provenance_graph_retriever,
)
from graph_memory.models.graph_retriever.text_embeddings import (
    PrecomputedGraphFeatureProvider,
)
from graph_memory.models.graph_retriever.training import train_graph_retriever
from graph_memory.stages.frozen_embeddings import FrozenEmbeddingStore
from graph_memory.training_pairs.contracts import TrainPairDataset


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
PROVENANCE_GRAPHS_ADAPTER = TypeAdapter(list[ProvenanceGraph])
ISETRACE_RANKINGS_ADAPTER = TypeAdapter(list[ISETraceRankingRecord])
ISETRACE_LABELS_ADAPTER = TypeAdapter(list[ISETraceLabelRecord])
TRAIN_PAIRS_ADAPTER = TypeAdapter(list[TrainPairRecord])


def materialize_dense_finetune_model(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    config: DenseFinetuneMethodConfig,
    train_prepared: DatasetArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    encoder_source: EncoderSourceRef,
    implementation_version: str,
) -> ModelArtifactRef:
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
    train_requests, train_compiled_labels, train_group_ids = _dense_finetune_split(
        dataset,
        train_tasks,
        train_labels,
        variant=config.variant,
    )
    dev_requests, dev_compiled_labels, _ = _dense_finetune_split(
        dataset,
        dev_tasks,
        dev_labels,
        variant=config.variant,
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
            "variant": config.variant,
            "prepared_digest": train_prepared.digest,
            "pairs_digest": train_pairs.digest,
            "dev_digest": dev_prepared.digest,
            "encoder_identity": immutable_source_identity(encoder_source),
            "implementation_version": implementation_version,
        },
    ) as publisher:
        trainer_output = publisher.workspace / "trainer_output"
        model_dir = publisher.workspace / "model"
        TrainPairDataset(
            requests=tuple(train_requests),
            labels=tuple(train_compiled_labels),
            pairs=tuple(pairs),
        )
        _validate_dev_split(dev_requests, dev_compiled_labels)
        _validate_optional_task_values(
            train_requests, train_group_ids, name="train group IDs"
        )
        settings = effective.train
        encoder = effective.encoder
        result = train_dense_finetune(
            config=DenseFinetuneRunConfig(
                variant=config.variant,
                base_model=encoder.model_name,
                query_prefix=encoder.query_prefix,
                passage_prefix=encoder.passage_prefix,
                batch_size=encoder.batch_size,
                data=settings.data,
                trainer=settings.trainer,
                selection=settings.selection,
            ),
            train_requests=train_requests,
            train_pairs=pairs,
            train_group_ids=train_group_ids,
            dev_requests=dev_requests,
            dev_labels=dev_compiled_labels,
            task_local_dev=dataset == "isetrace",
            output_dir=trainer_output,
            model_dir=model_dir,
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
                "variant": config.variant,
                "effective_sampling": train_pairs.origin.get("sampling_config"),
            },
        )
    assert isinstance(artifact, ModelArtifactRef)
    return artifact


def _dense_finetune_split(
    dataset: DatasetName,
    tasks: list[object],
    labels: list[object],
    *,
    variant: DenseCandidateView,
) -> tuple[
    list[TextRankingRequest],
    list[EvidenceLabel],
    dict[str, str],
]:
    if dataset != "isetrace":
        requests = text_ranking_requests_for_dataset(dataset, tasks)
        compiled_labels = evidence_labels_for_dataset(dataset, labels)
        return requests, compiled_labels, {}

    rankings = ISETRACE_RANKINGS_ADAPTER.validate_python(tasks)
    isetrace_labels = ISETRACE_LABELS_ADAPTER.validate_python(labels)
    adapter = (
        adapt_provenance_unit_dense_training_split
        if variant == "provenance_unit"
        else adapt_flat_dense_training_split
    )
    requests, compiled_labels = adapter(rankings, isetrace_labels)
    group_ids = {ranking.task_id: ranking.graph_id for ranking in rankings}
    return requests, compiled_labels, group_ids


def materialize_evidence_rgcn_model(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    method: str,
    variant: str,
    config: RgcnStageConfig,
    train_prepared: DatasetArtifactRef,
    train_graphs: EvidenceGraphArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    dev_graphs: EvidenceGraphArtifactRef,
    encoder_source: EncoderSourceRef,
    seed_model: ModelArtifactRef | None,
    frozen_embeddings: FrozenEmbeddingsArtifactRef,
    implementation_version: str,
) -> ModelArtifactRef:
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
    train_requests = text_ranking_requests_for_dataset(dataset, train_tasks)
    compiled_train_labels = evidence_labels_for_dataset(dataset, train_labels)
    dev_requests = text_ranking_requests_for_dataset(dataset, dev_tasks)
    compiled_dev_labels = evidence_labels_for_dataset(dataset, dev_labels)
    TrainPairDataset(
        requests=tuple(train_requests),
        labels=tuple(compiled_train_labels),
        graphs=tuple(train_graph_values),
        pairs=tuple(pair_values),
    )
    _validate_dev_split(dev_requests, compiled_dev_labels, dev_graph_values)
    seed_dir = (
        artifact_payload_path(seed_model, "model") if seed_model is not None else None
    )
    embedding_store = FrozenEmbeddingStore(frozen_embeddings)
    if embedding_store.index.family != "evidence":
        raise ValueError("Evidence R-GCN requires evidence frozen embeddings.")
    encoder_settings = _effective_rgcn_encoder(
        effective_encoder, seed_checkpoint=seed_dir
    )
    train_embeddings = embedding_store.partition("train")
    dev_embeddings = embedding_store.partition("dev")
    embedding_dim = embedding_store.index.embedding_dim
    train_provider = PrecomputedGraphFeatureProvider(
        train_embeddings, embedding_dim=embedding_dim
    )
    dev_provider = PrecomputedGraphFeatureProvider(
        dev_embeddings, embedding_dim=embedding_dim
    )
    settings = config.train
    model_config = default_model_config(
        method_name=method,
        encoder_model=encoder_settings.model_name,
        encoder_dim=embedding_dim,
        query_prefix=encoder_settings.query_prefix,
        passage_prefix=encoder_settings.passage_prefix,
        encoder_batch_size=encoder_settings.batch_size,
        hidden_dim=settings.model.hidden_dim,
        num_layers=settings.model.num_layers,
        dropout=settings.model.dropout,
        ablation_name=settings.model.ablation,
    )
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
            "encoder_identity": immutable_source_identity(encoder_source),
            "seed_model_digest": None if seed_model is None else seed_model.digest,
            "frozen_embeddings_digest": frozen_embeddings.digest,
            "implementation_version": implementation_version,
        },
    ) as publisher:
        result = train_graph_retriever(
            train_requests=train_requests,
            train_graphs=train_graph_values,
            train_labels=compiled_train_labels,
            train_pairs=pair_values,
            dev_requests=dev_requests,
            dev_labels=compiled_dev_labels,
            dev_graphs=dev_graph_values,
            model_config=model_config,
            training_config=settings.trainer,
            text_embedding_provider=train_provider,
            seed_signal_provider=train_provider,
            dev_text_embedding_provider=dev_provider,
            dev_seed_signal_provider=dev_provider,
            selection_settings=settings.selection,
            device=settings.trainer.device,
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
    return artifact


def materialize_provenance_rgcn_model(
    store: ProcessedAssetStore,
    *,
    config: ProvenanceRgcnMethodConfig,
    train_prepared: DatasetArtifactRef,
    train_pairs: TrainingPairsArtifactRef,
    dev_prepared: DatasetArtifactRef,
    encoder_source: EncoderSourceRef,
    seed_model: ModelArtifactRef | None,
    frozen_embeddings: FrozenEmbeddingsArtifactRef,
    implementation_version: str,
) -> ModelArtifactRef:
    if config.method != "provenance_rgcn":
        raise ValueError("provenance model stage requires method=provenance_rgcn")
    if (config.seed is None) != (seed_model is None):
        raise ValueError(
            "provenance R-GCN seed config and seed model must either both be present or both be absent"
        )
    effective_encoder = _resolved_encoder(config.encoder, encoder_source)
    seed_dir = (
        artifact_payload_path(seed_model, "model") if seed_model is not None else None
    )
    encoder_settings = _effective_rgcn_encoder(
        effective_encoder, seed_checkpoint=seed_dir
    )
    if config.seed is not None and seed_dir is not None:
        seed_metadata = load_dense_ft_model_metadata(seed_dir)
        if seed_metadata.variant != config.seed.variant:
            raise ValueError(
                "provenance R-GCN Dense-FT seed checkpoint variant does not match seed config"
            )
    train_requests, train_labels = _load_provenance_split(train_prepared)
    dev_requests, dev_labels = _load_provenance_split(dev_prepared)
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
        encoder_model=encoder_settings.model_name,
        encoder_dim=encoder_dim,
        query_prefix=encoder_settings.query_prefix,
        passage_prefix=encoder_settings.passage_prefix,
        encoder_batch_size=encoder_settings.batch_size,
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
            "encoder_identity": immutable_source_identity(encoder_source),
            "seed_model_digest": None if seed_model is None else seed_model.digest,
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
            model_config=model_config,
            training_config=config.train.trainer,
            text_embedding_provider=train_provider,
            seed_signal_provider=train_provider,
            dev_text_embedding_provider=dev_provider,
            dev_seed_signal_provider=dev_provider,
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
                "selection_metric": "dev_recall_at_5",
                "seed_model_digest": None if seed_model is None else seed_model.digest,
            },
        )
    assert isinstance(artifact, ModelArtifactRef)
    return artifact


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
    return adapt_provenance_training_split(rankings, labels, graphs)


def _effective_rgcn_encoder(
    encoder: DenseEncoderConfig,
    *,
    seed_checkpoint,
) -> DenseEncoderConfig:
    if seed_checkpoint is None:
        return encoder
    metadata = load_dense_ft_model_metadata(seed_checkpoint)
    return encoder.model_copy(
        update={
            "model_name": str(seed_checkpoint),
            "query_prefix": metadata.query_prefix,
            "passage_prefix": metadata.passage_prefix,
            "batch_size": metadata.batch_size,
        }
    )


def _validate_dev_split(
    requests: list[TextRankingRequest],
    labels: list[EvidenceLabel],
    graphs: list[EvidenceGraph] | None = None,
) -> None:
    request_ids = [request.task_id for request in requests]
    label_ids = [label.task_id for label in labels]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("dev request task IDs must be unique")
    if set(request_ids) != set(label_ids) or len(label_ids) != len(set(label_ids)):
        raise ValueError("dev requests and labels must align")
    if graphs:
        graph_ids = [graph.task_id for graph in graphs]
        if set(request_ids) != set(graph_ids) or len(graph_ids) != len(set(graph_ids)):
            raise ValueError("dev requests and graphs must align")


def _validate_optional_task_values(
    requests: list[TextRankingRequest],
    values: dict[str, str],
    *,
    name: str,
) -> None:
    if values and set(values) != {request.task_id for request in requests}:
        raise ValueError(f"dense-ft {name} must align with requests")


def _resolved_encoder(
    encoder: DenseEncoderConfig,
    source: EncoderSourceRef,
) -> DenseEncoderConfig:
    if isinstance(source, RevisionSourceRef):
        return encoder
    return encoder.model_copy(update={"model_name": source.uri})


__all__ = [
    "EncoderSourceRef",
    "materialize_dense_finetune_model",
    "materialize_evidence_rgcn_model",
    "materialize_provenance_rgcn_model",
]
