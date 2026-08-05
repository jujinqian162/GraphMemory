from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from graph_memory.graphs.contracts import EvidenceGraph, EvidenceGraphBatch
from graph_memory.datasets.selection import evidence_graph_build_requests_for_dataset
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


def build_evidence_graph_data(
    dataset: DatasetName,
    prepared: DatasetArtifactRef,
    config: GraphBuildConfig,
) -> tuple[list[EvidenceGraph], dict[str, JsonValue]]:
    tasks_path = artifact_payload_path(prepared, "tasks")
    ranking_records = read_json(tasks_path)
    requests = evidence_graph_build_requests_for_dataset(dataset, ranking_records)
    domain_config = DomainGraphBuildConfig(**config.model_dump())
    graphs = build_graphs(
        requests,
        domain_config,
        progress_desc=f"build evidence graphs ({dataset})",
    )
    expected_item_ids_by_task_id = {
        request.task_id: frozenset(node.node_id for node in request.nodes)
        for request in requests
    }
    EvidenceGraphBatch(
        graphs=tuple(graphs),
        expected_item_ids_by_task_id=expected_item_ids_by_task_id,
    )
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
) -> EvidenceGraphArtifactRef:
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
        write_json(
            publisher.workspace / "graphs.json",
            [
                graph.model_dump(mode="json", exclude_none=True)
                for graph in graphs
            ],
        )
        write_json(publisher.workspace / "statistics.json", statistics)
        artifact = publisher.publish(
            {"graphs": "graphs.json", "statistics": "statistics.json"},
            shape={"graphs": len(graphs)},
            metadata={"split": split},
        )
    assert isinstance(artifact, EvidenceGraphArtifactRef)
    return artifact


__all__ = ["build_evidence_graph_data", "materialize_evidence_graphs"]
