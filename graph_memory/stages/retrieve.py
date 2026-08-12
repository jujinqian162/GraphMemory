from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.provenance import ProvenanceGraph
from graph_memory.retrieval.results import RankedResult
from graph_memory.datasets.isetrace.benchmark_records import ISETraceRankingRecord
from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.embeddings import SentenceEncoder
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    DirectorySourceRef,
    EvidenceGraphArtifactRef,
    FileSourceRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
    ProcessedAssetStore,
    RevisionSourceRef,
    artifact_payload_path,
    immutable_source_identity,
)
from graph_memory.experiment.config import (
    CrossEncoderMethodConfig,
    DatasetName,
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    DenseMethodConfig,
    MethodConfig,
    ProvenancePathMethodConfig,
    ProvenanceRgcnMethodConfig,
    RgcnMethodConfig,
)
from graph_memory.io import read_json, write_json
from graph_memory.registry.retrieval_builders import (
    RetrievalProvenance,
    build_retrieval,
)
from graph_memory.retrieval.execution.service import run_retrieval


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
PROVENANCE_GRAPHS_ADAPTER = TypeAdapter(list[ProvenanceGraph])
ISETRACE_RANKINGS_ADAPTER = TypeAdapter(list[ISETraceRankingRecord])


def run_retrieve_stage(
    method: MethodConfig,
    *,
    dataset: DatasetName,
    top_k: int,
    task_inputs: Sequence[object],
    evidence_graphs: list[EvidenceGraph] | None,
    model: ModelArtifactRef | None,
    encoder_source: EncoderSourceRef | None,
    device: str,
    dense_encoder: SentenceEncoder | None = None,
    provenance_graphs: list[ProvenanceGraph] | None = None,
) -> tuple[list[RankedResult], RetrievalProvenance]:
    text_requests = text_ranking_requests_for_dataset(
        dataset,
        task_inputs,
        isetrace_representation=(
            "provenance"
            if (
                isinstance(
                    method,
                    (
                        DenseMethodConfig,
                        DenseFinetuneMethodConfig,
                        CrossEncoderMethodConfig,
                    ),
                )
                and method.variant == "provenance_unit"
            )
            or isinstance(
                method,
                (ProvenancePathMethodConfig, ProvenanceRgcnMethodConfig),
            )
            else "flat"
        ),
    )
    graph_ids_by_task_id = None
    if isinstance(method, (ProvenancePathMethodConfig, ProvenanceRgcnMethodConfig)):
        records = ISETRACE_RANKINGS_ADAPTER.validate_python(task_inputs)
        graph_ids_by_task_id = {
            record.task_id: record.graph_id for record in records
        }
    retrieval_method, retrieval_provenance, execution_requests = build_retrieval(
        method,
        text_requests=text_requests,
        device=device,
        encoder_model_name=_encoder_model_name(encoder_source),
        checkpoint=_model_checkpoint(method, model),
        evidence_graphs=evidence_graphs,
        provenance_graphs=provenance_graphs,
        graph_ids_by_task_id=graph_ids_by_task_id,
        dense_encoder=dense_encoder,
    )
    predictions = run_retrieval(
        retrieval_method=retrieval_method,
        requests=execution_requests,
        top_k=top_k,
    )
    return predictions, retrieval_provenance


def materialize_rankings(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    method: MethodConfig,
    top_k: int,
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    model: ModelArtifactRef | None,
    encoder_source: EncoderSourceRef | None,
    device: str,
    implementation_version: str,
) -> PredictionsArtifactRef:
    # Prepared tasks already passed dataset validation at materialize_prepared_split.
    task_inputs = read_json(artifact_payload_path(prepared, "tasks"))
    graph_values = (
        EVIDENCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(evidence_graphs, "graphs"))
        )
        if evidence_graphs is not None
        else []
    )
    provenance_graph_values = (
        PROVENANCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(prepared, "provenance_graphs"))
        )
        if isinstance(method, (ProvenancePathMethodConfig, ProvenanceRgcnMethodConfig))
        else []
    )
    started = time.perf_counter()
    predictions, retrieval_provenance = run_retrieve_stage(
        method,
        dataset=dataset,
        top_k=top_k,
        task_inputs=task_inputs,
        evidence_graphs=graph_values,
        provenance_graphs=provenance_graph_values,
        model=model,
        encoder_source=encoder_source,
        device=device,
    )
    production_seconds = time.perf_counter() - started
    provenance = retrieval_provenance
    method_name = method.method
    variant = getattr(method, "variant", None)
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.PREDICTIONS,
        namespace=method_name,
        task_identity=f"rank-{method_name}",
        origin={
            "stage": "rank",
            "dataset": dataset,
            "method": method_name,
            "variant": variant,
            "method_config": method.model_dump(mode="json"),
            "prepared_digest": prepared.digest,
            "graph_digest": None if evidence_graphs is None else evidence_graphs.digest,
            "model_digest": None if model is None else model.digest,
            "encoder_identity": (
                None if encoder_source is None else immutable_source_identity(encoder_source)
            ),
            "implementation_version": implementation_version,
        },
    ) as publisher:
        write_json(
            publisher.workspace / "predictions.json",
            [
                prediction.model_dump(mode="json", exclude_none=True)
                for prediction in predictions
            ],
        )
        write_json(publisher.workspace / "provenance.json", provenance)
        artifact = publisher.publish(
            {
                "predictions": "predictions.json",
                "provenance": "provenance.json",
            },
            shape={"predictions": len(predictions)},
            metadata={"production_seconds": production_seconds},
        )
    assert isinstance(artifact, PredictionsArtifactRef)
    return artifact


def _model_checkpoint(
    method: MethodConfig,
    model: ModelArtifactRef | None,
) -> Path | None:
    if isinstance(method, (DenseFinetuneMethodConfig, CrossEncoderMethodConfig)):
        return _model_payload(model, "model")
    if isinstance(
        method,
        (ProvenanceRgcnMethodConfig, RgcnMethodConfig, DenseFtRgcnMethodConfig),
    ):
        return _model_payload(model, "checkpoint")
    return None


def _encoder_model_name(source: EncoderSourceRef | None) -> str | None:
    if isinstance(source, (FileSourceRef, DirectorySourceRef)):
        return source.uri
    return None


def _model_payload(model: ModelArtifactRef | None, role: str) -> Path:
    if model is None:
        raise ValueError(f"trainable retrieval requires model payload role={role}")
    return artifact_payload_path(model, role)


__all__ = ["materialize_rankings", "run_retrieve_stage"]
