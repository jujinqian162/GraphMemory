from __future__ import annotations

import shutil
from typing import cast

from pydantic import JsonValue, TypeAdapter

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.datasets.selection import (
    evidence_labels_for_dataset,
    execution_provenance_requests_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
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
    ProvenanceRgcnStageConfig,
    RgcnTrainStageConfig,
)
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.provenance_rgcn import save_provenance_rgcn_checkpoint
from graph_memory.stages.results import ModelResult
from graph_memory.stages.train_payloads import (
    DenseFinetuneTrainPayload,
    ProvenanceRgcnTrainPayload,
    RgcnTrainPayload,
)
from graph_memory.stages.trainers import (
    DenseFinetuneMethodTrainer,
    ProvenanceRgcnMethodTrainer,
    RgcnGraphRetrieverTrainer,
)


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
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
                train_requests=tuple(
                    text_ranking_requests_for_dataset(dataset, train_tasks)
                ),
                train_labels=tuple(
                    evidence_labels_for_dataset(dataset, train_labels)
                ),
                train_pairs=tuple(pairs),
                dev_requests=tuple(
                    text_ranking_requests_for_dataset(dataset, dev_tasks)
                ),
                dev_labels=tuple(
                    evidence_labels_for_dataset(dataset, dev_labels)
                ),
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
        },
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
            "implementation_version": implementation_version,
        },
    ) as publisher:
        result = RgcnGraphRetrieverTrainer(
            method=method,
            encoder=effective_encoder,
            train_config=config.train,
            seed_checkpoint=seed_dir,
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
    dataset: DatasetName,
    config: ProvenanceRgcnStageConfig,
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
    pair_summary = cast(
        dict[str, object],
        read_json(artifact_payload_path(train_pairs, "summary")),
    )
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.MODEL,
        namespace=config.method,
        task_identity=f"train-{config.method}",
        origin={
            "stage": "train",
            "dataset": dataset,
            "method": config.method,
            "variant": config.variant,
            "prepared_digest": train_prepared.digest,
            "pairs_digest": train_pairs.digest,
            "dev_digest": dev_prepared.digest,
            "encoder_identity": _encoder_identity(encoder_source),
            "implementation_version": implementation_version,
        },
    ) as publisher:
        result = ProvenanceRgcnMethodTrainer(effective).train(
            ProvenanceRgcnTrainPayload(
                train_requests=tuple(
                    execution_provenance_requests_for_dataset(dataset, train_tasks)
                ),
                train_labels=tuple(
                    evidence_labels_for_dataset(dataset, train_labels)
                ),
                train_pairs=tuple(pairs),
                dev_requests=tuple(
                    execution_provenance_requests_for_dataset(dataset, dev_tasks)
                ),
                dev_labels=tuple(
                    evidence_labels_for_dataset(dataset, dev_labels)
                ),
            )
        )
        checkpoints = publisher.workspace / "checkpoints"
        epoch_checkpoint = checkpoints / f"checkpoint_epoch_{result.best_epoch}.pt"
        first_train_task = train_tasks[0] if train_tasks else {}
        train_metadata = (
            first_train_task.get("metadata", {})
            if isinstance(first_train_task, dict)
            else {}
        )
        construction_identity = (
            train_metadata.get("graph_construction", "missing")
            if isinstance(train_metadata, dict)
            else "missing"
        )
        best_metrics = result.best_metrics
        for path in (epoch_checkpoint, checkpoints / "best.pt"):
            save_provenance_rgcn_checkpoint(
                path,
                method_name=config.method,
                model=result.model,
                optimizer_state_dict=result.optimizer_state_dict,
                epoch=result.best_epoch,
                global_step=result.global_step,
                best_dev_metric=result.best_dev_metric,
                model_config=result.model_config,
                training_config=result.training_config,
                effective_variant=config.variant,
                scientific_identity={
                    "dataset": {
                        "name": dataset,
                        "schema_version": 3,
                        "train_prepared_digest": train_prepared.digest,
                        "dev_prepared_digest": dev_prepared.digest,
                    },
                    "construction": construction_identity,
                    "pairs": {
                        "digest": train_pairs.digest,
                        "sampling": pair_summary.get("sampling_config", {}),
                    },
                    "encoder": _encoder_identity(encoder_source),
                },
                best_metrics=best_metrics,
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
            },
        )
    assert isinstance(artifact, ModelArtifactRef)
    return ModelResult(
        method=config.method,
        artifact=artifact,
        training_history=history,
        metadata={
            "variant": config.variant,
            "best_epoch": result.best_epoch,
            "best_dev_metric": result.best_dev_metric,
        },
    )


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
