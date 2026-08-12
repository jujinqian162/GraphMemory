from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue, TypeAdapter

from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.graphs.provenance import ProvenanceGraph
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
    PairBuildConfig,
)
from graph_memory.io import read_json, write_json
from graph_memory.models.graph_retriever.provenance import provenance_train_pair_task
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.training_pairs.requests import TrainPairBuildTask


EncoderSourceRef = FileSourceRef | DirectorySourceRef | RevisionSourceRef
EVIDENCE_GRAPHS_ADAPTER = TypeAdapter(list[EvidenceGraph])
PROVENANCE_GRAPHS_ADAPTER = TypeAdapter(list[ProvenanceGraph])
ISETRACE_RANKINGS_ADAPTER = TypeAdapter(list[ISETraceRankingRecord])
ISETRACE_LABELS_ADAPTER = TypeAdapter(list[ISETraceLabelRecord])


def build_training_pair_data(
    dataset: DatasetName,
    prepared: DatasetArtifactRef,
    *,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    config: PairBuildConfig,
    encoder_source: EncoderSourceRef,
) -> tuple[list[TrainPairRecord], dict[str, JsonValue]]:
    tasks = cast(
        list[Mapping[str, object]],
        read_json(artifact_payload_path(prepared, "tasks")),
    )
    labels = cast(list[object], read_json(artifact_payload_path(prepared, "labels")))
    graphs = (
        EVIDENCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(evidence_graphs, "graphs"))
        )
        if evidence_graphs is not None
        else []
    )
    dense_config = _dense_config(
        config.sampling,
        encoder=config.encoder,
        encoder_source=encoder_source,
        device=config.device,
    )
    provenance_graphs = (
        PROVENANCE_GRAPHS_ADAPTER.validate_python(
            read_json(artifact_payload_path(prepared, "provenance_graphs"))
        )
        if dataset == "isetrace" and config.method == "provenance_rgcn"
        else []
    )
    result = build_train_pairs(
        _pair_tasks(
            dataset,
            config.method,
            tasks,
            labels,
            graphs,
            candidate_view=config.candidate_view,
            provenance_graphs=provenance_graphs,
        ),
        config.sampling,
        dense_config=dense_config,
        progress_desc="build training pairs",
    )
    return list(result.pairs), cast(
        dict[str, JsonValue],
        result.summary.model_dump(mode="json", exclude_none=True),
    )


def materialize_training_pairs(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    prepared: DatasetArtifactRef,
    evidence_graphs: EvidenceGraphArtifactRef | None,
    config: PairBuildConfig,
    encoder_source: EncoderSourceRef,
    implementation_version: str,
) -> TrainingPairsArtifactRef:
    pairs, summary = build_training_pair_data(
        dataset,
        prepared,
        evidence_graphs=evidence_graphs,
        config=config,
        encoder_source=encoder_source,
    )
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.TRAINING_PAIRS,
        namespace=dataset,
        task_identity=f"pairs-{dataset}",
        origin={
            "stage": "pairs",
            "dataset": dataset,
            "method": config.method,
            "candidate_view": config.candidate_view,
            "sampling_config": config.sampling.model_dump(mode="json"),
            "prepared_digest": prepared.digest,
            "graph_digest": None if evidence_graphs is None else evidence_graphs.digest,
            "encoder_identity": immutable_source_identity(encoder_source),
            "implementation_version": implementation_version,
        },
    ) as publisher:
        write_json(
            publisher.workspace / "pairs.json",
            [pair.model_dump(mode="json", exclude_none=True) for pair in pairs],
        )
        write_json(publisher.workspace / "summary.json", summary)
        artifact = publisher.publish(
            {"pairs": "pairs.json", "summary": "summary.json"},
            shape={"pairs": len(pairs)},
        )
    assert isinstance(artifact, TrainingPairsArtifactRef)
    return artifact


def _pair_tasks(
    dataset: DatasetName,
    method: str,
    task_inputs: list[Mapping[str, object]],
    labels: list[object],
    graphs: list[EvidenceGraph],
    *,
    candidate_view: DenseCandidateView,
    provenance_graphs: list[ProvenanceGraph],
) -> list[TrainPairBuildTask]:
    if dataset == "isetrace":
        rankings = ISETRACE_RANKINGS_ADAPTER.validate_python(task_inputs)
        isetrace_labels = ISETRACE_LABELS_ADAPTER.validate_python(labels)
        if method in {"dense_ft", "cross_encoder"}:
            adapter = (
                adapt_provenance_unit_dense_training_split
                if candidate_view == "provenance_unit"
                else adapt_flat_dense_training_split
            )
            requests, compiled_labels = adapter(rankings, isetrace_labels)
            return [
                TrainPairBuildTask(text_request=request, label=label)
                for request, label in zip(requests, compiled_labels, strict=True)
            ]
        if method != "provenance_rgcn":
            raise ValueError(
                f"dataset='isetrace' has no train-pair projection for method={method!r}"
            )
        requests, compiled_labels = adapt_provenance_training_split(
            rankings,
            isetrace_labels,
            provenance_graphs,
        )
        return [
            provenance_train_pair_task(request, label)
            for request, label in zip(requests, compiled_labels, strict=True)
        ]

    text_requests = {
        request.task_id: request
        for request in text_ranking_requests_for_dataset(dataset, task_inputs)
    }
    labels_by_task_id = {
        label.task_id: label for label in evidence_labels_for_dataset(dataset, labels)
    }
    graphs_by_task_id = {graph.task_id: graph for graph in graphs}
    return [
        TrainPairBuildTask(
            text_request=request,
            label=labels_by_task_id[task_id],
            graph=graphs_by_task_id.get(task_id),
        )
        for task_id, request in text_requests.items()
    ]


def _dense_config(
    sampling: NegativeSamplingConfig,
    *,
    encoder: DenseEncoderConfig,
    encoder_source: EncoderSourceRef,
    device: str,
) -> DenseConfig | None:
    if sampling.hard_dense_per_positive <= 0:
        return None
    model_name = (
        encoder_source.uri
        if isinstance(encoder_source, (FileSourceRef, DirectorySourceRef))
        else encoder.model_name
    )
    return DenseConfig(
        model_name=model_name,
        query_prefix=encoder.query_prefix,
        passage_prefix=encoder.passage_prefix,
        batch_size=encoder.batch_size,
        device=device,
    )


__all__ = [
    "EncoderSourceRef",
    "build_training_pair_data",
    "materialize_training_pairs",
]
