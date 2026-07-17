from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.datasets.selection import (
    evidence_graph_build_requests_for_dataset,
    validate_ranking_records_for_dataset,
)
from graph_memory.experiment.artifacts import (
    ArtifactKind,
    ArtifactPublisher,
    DatasetArtifactRef,
    EvidenceGraphArtifactRef,
    ProcessedAssetStore,
    artifact_payload_path,
)
from graph_memory.experiment.config import (
    DatasetName,
    GraphBuildConfig,
    SplitName,
)
from graph_memory.graphs.config import GraphBuildConfig as DomainGraphBuildConfig
from graph_memory.graphs.construction.builder import build_graphs
from graph_memory.graphs.statistics import graph_statistics
from graph_memory.io import read_json, write_json
from graph_memory.stages.results import EvidenceGraphResult
from graph_memory.validation import validate_graphs


def build_evidence_graph_data(
    dataset: DatasetName,
    prepared: DatasetArtifactRef,
    config: GraphBuildConfig,
) -> tuple[list[EvidenceGraph], dict[str, JsonValue]]:
    tasks_path = artifact_payload_path(prepared, "tasks")
    ranking_records = read_json(tasks_path)
    validate_ranking_records_for_dataset(dataset, ranking_records)
    requests = evidence_graph_build_requests_for_dataset(dataset, ranking_records)
    domain_config = DomainGraphBuildConfig(**config.model_dump())
    graphs = build_graphs(requests, domain_config)
    validate_graphs(graphs, requests)
    statistics = graph_statistics(
        graphs,
        graph_config={"dataset": dataset, **config.model_dump(mode="json")},
    )
    return graphs, cast(dict[str, JsonValue], dict(statistics))


def materialize_evidence_graphs(
    store: ProcessedAssetStore,
    *,
    dataset: DatasetName,
    split: SplitName,
    prepared: DatasetArtifactRef,
    config: GraphBuildConfig,
    implementation_version: str,
) -> EvidenceGraphResult:
    graphs, statistics = build_evidence_graph_data(dataset, prepared, config)
    with ArtifactPublisher(
        store,
        kind=ArtifactKind.EVIDENCE_GRAPH,
        namespace=dataset,
        task_identity=f"graphs-{dataset}-{split}",
        origin={
            "stage": "evidence_graphs",
            "dataset": dataset,
            "split": split,
            "prepared_digest": prepared.digest,
            "implementation_version": implementation_version,
        },
    ) as publisher:
        write_json(publisher.workspace / "graphs.json", graphs)
        write_json(publisher.workspace / "statistics.json", statistics)
        artifact = publisher.publish(
            {"graphs": "graphs.json", "statistics": "statistics.json"},
            shape={"graphs": len(graphs)},
            metadata={"split": split},
        )
    assert isinstance(artifact, EvidenceGraphArtifactRef)
    return EvidenceGraphResult(
        split=split,
        artifact=artifact,
        statistics=statistics,
    )


__all__ = ["build_evidence_graph_data", "materialize_evidence_graphs"]
