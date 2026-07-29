from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pydantic import JsonValue, TypeAdapter

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.results import RankedResult
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
    DenseMethodConfig,
    GraphRAGMethodConfig,
    RankingMethodConfig,
    TrainableRankingConfig,
)
from graph_memory.io import read_json, write_json
from graph_memory.registry import Registry
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
    RetrievalMethodId,
    RetrievalProvenance,
    RetrievalTaskFamily,
)
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.stages.results import RankingResult


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])


@dataclass(frozen=True)
class RetrieveStageResult:
    predictions: list[RankedResult]
    provenance: RetrievalProvenance


def run_retrieve_stage(
    method: RankingMethodConfig,
    *,
    dataset: DatasetName,
    top_k: int,
    task_inputs: Sequence[object],
    evidence_graphs: list[EvidenceGraph] | None,
    model: ModelArtifactRef | None,
    encoder_source: EncoderSourceRef | None,
    device: str,
    dense_encoder: SentenceEncoder | None = None,
) -> RetrieveStageResult:
    text_requests = text_ranking_requests_for_dataset(dataset, task_inputs)
    settings = _retrieval_settings(
        method,
        top_k=top_k,
        model=model,
        encoder_source=encoder_source,
        device=device,
    )
    built = Registry.retrieval.build(
        settings,
        _build_payload(
            method,
            dataset=dataset,
            text_requests=text_requests,
            evidence_graphs=evidence_graphs or [],
            dense_encoder=dense_encoder,
        ),
    )
    predictions = run_retrieval(
        retrieval_method=built.method,
        tasks=built.execution_tasks,
        top_k=top_k,
    )
    return RetrieveStageResult(predictions=predictions, provenance=built.provenance)


def materialize_rankings(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    method: RankingMethodConfig,
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
    started = time.perf_counter()
    result = run_retrieve_stage(
        method,
        dataset=dataset,
        top_k=top_k,
        task_inputs=task_inputs,
        evidence_graphs=graph_values,
        model=model,
        encoder_source=encoder_source,
        device=device,
    )
    production_seconds = time.perf_counter() - started
    provenance = _provenance_json(result.provenance)
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
                for prediction in result.predictions
            ],
        )
        write_json(publisher.workspace / "provenance.json", provenance)
        artifact = publisher.publish(
            {
                "predictions": "predictions.json",
                "provenance": "provenance.json",
            },
            shape={"predictions": len(result.predictions)},
            metadata={"production_seconds": production_seconds},
        )
    assert isinstance(artifact, PredictionsArtifactRef)
    return RankingResult(
        method=method_name,
        artifact=artifact,
        provenance=provenance,
        production_seconds=production_seconds,
    )


def _build_payload(
    method: RankingMethodConfig,
    *,
    dataset: DatasetName,
    text_requests: list[TextRankingRequest],
    evidence_graphs: list[EvidenceGraph],
    dense_encoder: SentenceEncoder | None,
) -> object:
    if isinstance(method, (Bm25MethodConfig, DenseMethodConfig)) or (
        isinstance(method, TrainableRankingConfig) and method.method == "dense_ft"
    ):
        return FlatRetrievalBuildPayload(
            text_requests=text_requests,
            task_family=_task_family(dataset),
            dense_encoder=dense_encoder,
        )
    if isinstance(method, GraphRAGMethodConfig):
        return GraphRAGBuildPayload(
            text_requests=text_requests,
            task_family=_task_family(dataset),
            dense_encoder=dense_encoder,
        )
    if isinstance(method, TrainableRankingConfig) and method.method in {
        "dense_rgcn_graph_retriever",
        "dense_ft_rgcn_graph_retriever",
    }:
        return EvidenceRgcnBuildPayload(
            text_requests=text_requests,
            evidence_graphs=evidence_graphs,
            dense_encoder=dense_encoder,
        )
    raise TypeError(f"unsupported method config={type(method).__name__}")


def _retrieval_settings(
    method: RankingMethodConfig,
    *,
    top_k: int,
    model: ModelArtifactRef | None,
    encoder_source: EncoderSourceRef | None,
    device: str,
):
    if isinstance(method, Bm25MethodConfig):
        return Bm25RetrievalSettings(top_k=top_k)
    if isinstance(method, DenseMethodConfig):
        return DenseRetrievalSettings(
            top_k=top_k,
            encoder=_encoder_settings(method.encoder, encoder_source),
            device=device,
        )
    if isinstance(method, GraphRAGMethodConfig):
        return GraphRAGRetrievalSettings(
            top_k=top_k,
            encoder=_encoder_settings(method.encoder, encoder_source),
            config=method,
            device=device,
        )
    if isinstance(method, TrainableRankingConfig):
        if method.method == "dense_ft":
            return DenseFinetunedRetrievalSettings(
                top_k=top_k,
                checkpoint=_model_payload(model, "model"),
                device=device,
            )
        if method.method == "dense_rgcn_graph_retriever":
            return EvidenceRgcnRetrievalSettings(
                top_k=top_k,
                checkpoint=_model_payload(model, "checkpoint"),
                device=device,
            )
        if method.method == "dense_ft_rgcn_graph_retriever":
            return EvidenceRgcnRetrievalSettings(
                method=RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER,
                top_k=top_k,
                checkpoint=_model_payload(model, "checkpoint"),
                device=device,
            )
        raise TypeError(f"unsupported trainable method={method.method!r}")
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


def _task_family(dataset: DatasetName) -> RetrievalTaskFamily:
    del dataset
    return RetrievalTaskFamily.EVIDENCE_RETRIEVAL


def _provenance_json(provenance: RetrievalProvenance) -> dict[str, JsonValue]:
    value = asdict(provenance)
    value["method"] = provenance.method.value
    value["model"] = None if provenance.model is None else provenance.model.as_posix()
    return cast(dict[str, JsonValue], value)


def _encoder_identity(source: EncoderSourceRef) -> str:
    if isinstance(source, (FileSourceRef, DirectorySourceRef)):
        return source.digest
    return source.revision


__all__ = ["RetrieveStageResult", "materialize_rankings", "run_retrieve_stage"]
