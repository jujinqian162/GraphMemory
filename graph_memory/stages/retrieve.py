from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from pydantic import JsonValue, TypeAdapter

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
)
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    DatasetName,
    DenseEncoderConfig,
    DenseFinetuneMethodConfig,
    DenseFtRgcnMethodConfig,
    DenseMethodConfig,
    GraphRAGMethodConfig,
    MethodConfig,
    ProvenancePathMethodConfig,
    ProvenanceRgcnMethodConfig,
    RgcnMethodConfig,
)
from graph_memory.io import read_json, write_json
from graph_memory.registry.retrieval_builders import build_retrieval
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
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
    RetrievalMethodId,
    RetrievalProvenance,
)
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.stages.results import RankingResult


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
            if isinstance(method, ProvenancePathMethodConfig)
            or (
                isinstance(method, ProvenanceRgcnMethodConfig)
            )
            else "flat"
        ),
    )
    settings = _retrieval_settings(
        method,
        top_k=top_k,
        model=model,
        encoder_source=encoder_source,
        device=device,
    )
    retrieval_method, retrieval_provenance, execution_requests = build_retrieval(
        settings,
        _build_payload(
            method,
            dataset=dataset,
            text_requests=text_requests,
            evidence_graphs=evidence_graphs or [],
            provenance_graphs=provenance_graphs or [],
            task_inputs=task_inputs,
            dense_encoder=dense_encoder,
        ),
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
) -> RankingResult:
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
    provenance = _provenance_json(retrieval_provenance)
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
                None if encoder_source is None else _encoder_identity(encoder_source)
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
    return RankingResult(
        artifact=artifact,
        production_seconds=production_seconds,
    )


def _build_payload(
    method: MethodConfig,
    *,
    dataset: DatasetName,
    text_requests: list[TextRankingRequest],
    evidence_graphs: list[EvidenceGraph],
    provenance_graphs: list[ProvenanceGraph],
    task_inputs: Sequence[object],
    dense_encoder: SentenceEncoder | None,
) -> object:
    if isinstance(method, (Bm25MethodConfig, DenseMethodConfig)) or (
        isinstance(method, DenseFinetuneMethodConfig)
    ):
        return FlatRetrievalBuildPayload(
            text_requests=text_requests,
            dense_encoder=dense_encoder,
        )
    if isinstance(method, GraphRAGMethodConfig):
        return GraphRAGBuildPayload(
            text_requests=text_requests,
            dense_encoder=dense_encoder,
        )
    if isinstance(method, ProvenancePathMethodConfig):
        records = ISETRACE_RANKINGS_ADAPTER.validate_python(task_inputs)
        return ProvenancePathBuildPayload(
            text_requests=text_requests,
            provenance_graphs=provenance_graphs,
            graph_ids_by_task_id={
                record.task_id: record.graph_id for record in records
            },
            dense_encoder=dense_encoder,
        )
    if isinstance(method, ProvenanceRgcnMethodConfig):
        records = ISETRACE_RANKINGS_ADAPTER.validate_python(task_inputs)
        return ProvenanceRgcnBuildPayload(
            text_requests=text_requests,
            provenance_graphs=provenance_graphs,
            graph_ids_by_task_id={
                record.task_id: record.graph_id for record in records
            },
            dense_encoder=dense_encoder,
        )
    if isinstance(method, (RgcnMethodConfig, DenseFtRgcnMethodConfig)):
        return EvidenceRgcnBuildPayload(
            text_requests=text_requests,
            evidence_graphs=evidence_graphs,
            dense_encoder=dense_encoder,
        )
    raise TypeError(f"unsupported method config={type(method).__name__}")


def _retrieval_settings(
    method: MethodConfig,
    *,
    top_k: int,
    model: ModelArtifactRef | None,
    encoder_source: EncoderSourceRef | None,
    device: str,
):
    if isinstance(method, Bm25MethodConfig):
        return Bm25RetrievalSettings()
    if isinstance(method, DenseMethodConfig):
        return DenseRetrievalSettings(
            encoder=_encoder_settings(method.encoder, encoder_source),
            device=device,
        )
    if isinstance(method, GraphRAGMethodConfig):
        return GraphRAGRetrievalSettings(
            encoder=_encoder_settings(method.encoder, encoder_source),
            config=method,
            device=device,
        )
    if isinstance(method, ProvenancePathMethodConfig):
        return ProvenancePathRetrievalSettings(
            encoder=_encoder_settings(method.encoder, encoder_source),
            config=method.retrieval_config(),
            device=device,
        )
    if isinstance(method, DenseFinetuneMethodConfig):
        return DenseFinetunedRetrievalSettings(
            checkpoint=_model_payload(model, "model"),
            device=device,
        )
    if isinstance(method, ProvenanceRgcnMethodConfig):
        return ProvenanceRgcnRetrievalSettings(
            checkpoint=_model_payload(model, "checkpoint"),
            device=device,
        )
    if isinstance(method, RgcnMethodConfig):
        return EvidenceRgcnRetrievalSettings(
            checkpoint=_model_payload(model, "checkpoint"),
            device=device,
        )
    if isinstance(method, DenseFtRgcnMethodConfig):
        return EvidenceRgcnRetrievalSettings(
            method=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
            checkpoint=_model_payload(model, "checkpoint"),
            device=device,
        )
    raise TypeError(f"unsupported method config={type(method).__name__}")


def _model_payload(model: ModelArtifactRef | None, role: str) -> Path:
    if model is None:
        raise ValueError(f"trainable retrieval requires model payload role={role}")
    return artifact_payload_path(model, role)


def _encoder_settings(
    config: DenseEncoderConfig,
    source: EncoderSourceRef | None,
) -> DenseEncoderSettings:
    model_name = config.model_name
    if isinstance(source, (FileSourceRef, DirectorySourceRef)):
        model_name = source.uri
    return DenseEncoderSettings(
        model_name=model_name,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        batch_size=config.batch_size,
    )


def _provenance_json(provenance: RetrievalProvenance) -> dict[str, JsonValue]:
    value = asdict(provenance)
    value["method"] = provenance.method.value
    value["model"] = None if provenance.model is None else provenance.model.as_posix()
    return cast(dict[str, JsonValue], value)


def _encoder_identity(source: EncoderSourceRef) -> str:
    if isinstance(source, (FileSourceRef, DirectorySourceRef)):
        return source.digest
    return source.revision


__all__ = ["materialize_rankings", "run_retrieve_stage"]
