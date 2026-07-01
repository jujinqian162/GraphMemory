from __future__ import annotations

from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricTableRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord
from graph_memory.evaluation.connectivity import GraphConnectivity
from graph_memory.evaluation.failure_cases import build_failure_cases
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.service import evaluate_results
from graph_memory.evaluation.tables import (
    EFFICIENCY_RESULT_COLUMNS,
    MAIN_RESULT_COLUMNS,
    PATH_RESULT_COLUMNS,
    WIDE_METRIC_COLUMNS,
    split_metric_tables,
)


def _evidence_labels(labels: list[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
        )
        for label in labels
    ]


def _evaluation_request(
    predictions: list[RankedResult],
    labels: list[HotpotQALabelRecord],
    graphs: list[MemoryGraph],
) -> EvidenceEvaluationRequest:
    return EvidenceEvaluationRequest(predictions=predictions, labels=_evidence_labels(labels), graphs=graphs)


def test_evaluate_results_emits_full_metric_row_with_efficiency_and_path_columns() -> None:
    predictions, labels, graphs = _evaluation_fixture()

    rows = evaluate_results(_evaluation_request(predictions, labels, graphs))

    assert rows == [
        {
            "Method": "bm25",
            "Recall@2": 0.5,
            "Recall@5": 1.0,
            "Recall@10": 1.0,
            "Evidence F1@5": 0.5714285714285715,
            "Evidence F1@10": 0.33333333333333337,
            "Full Support@5": 1.0,
            "Full Support@10": 1.0,
            "MRR": 1.0,
            "Connected Evidence Recall@5": 1.0,
            "Connected Evidence Recall@10": 1.0,
            "Query-Evidence Connectivity@10": 1.0,
            "Path Recall@10": "N/A",
            "Edge Recall@10": "N/A",
            "Retrieval Latency / Query": 7.0,
            "Index Build Time": 0.0,
            "Graph Construction Time": 0.0,
            "Memory Size": 4.0,
            "Avg Retrieved Nodes": 3.0,
            "Avg Retrieved Edges": 2.0,
        }
    ]


def test_split_metric_tables_partitions_columns_into_main_path_and_efficiency() -> None:
    predictions, labels, graphs = _evaluation_fixture()
    rows = evaluate_results(_evaluation_request(predictions, labels, graphs))

    main_rows, path_rows, efficiency_rows = split_metric_tables(cast(list[MetricTableRow], rows))

    assert list(main_rows[0]) == MAIN_RESULT_COLUMNS
    assert list(path_rows[0]) == PATH_RESULT_COLUMNS
    assert list(efficiency_rows[0]) == EFFICIENCY_RESULT_COLUMNS
    assert WIDE_METRIC_COLUMNS == [
        *MAIN_RESULT_COLUMNS,
        "Connected Evidence Recall@5",
        "Connected Evidence Recall@10",
        "Query-Evidence Connectivity@10",
        "Path Recall@10",
        "Edge Recall@10",
        "Retrieval Latency / Query",
        "Index Build Time",
        "Graph Construction Time",
        "Memory Size",
        "Avg Retrieved Nodes",
        "Avg Retrieved Edges",
    ]


def test_graph_connectivity_reports_directed_and_undirected_reachability() -> None:
    _, _, graphs = _evaluation_fixture()
    connectivity = GraphConnectivity.from_graph(graphs[0], allowed_nodes={"q", "m0", "m1", "m2"})

    assert connectivity.undirected_reachable("m0") == {"q", "m0", "m1", "m2"}
    assert connectivity.directed_reachable("q") == {"q", "m0", "m1", "m2"}


def test_build_failure_cases_reports_missing_gold_and_disconnected_evidence() -> None:
    _, labels, graphs = _evaluation_fixture()
    failure_prediction = cast(
        RankedResult,
        cast(
            object,
            {
                "task_id": "hotpot_eval_1",
                "method": "bm25",
                "ranked_nodes": [
                    {"node_id": "m0", "score": 3.0},
                    {"node_id": "m1", "score": 2.0},
                    {"node_id": "m2", "score": 1.0},
                ],
                "retrieved_subgraph": {"nodes": ["m0", "m1", "m2"], "edges": []},
            },
        ),
    )

    assert build_failure_cases(_evaluation_request([failure_prediction], labels, graphs), top_k=3, limit=1) == [
        {
            "debug_type": "failure_case",
            "task_id": "hotpot_eval_1",
            "method": "bm25",
            "failure_type": "missing_full_support_at_3",
            "gold_evidence_item_ids": ["m0", "m3"],
            "retrieved_top_k": ["m0", "m1", "m2"],
            "missing_gold_nodes": ["m3"],
            "connected_gold_in_top_k": False,
        }
    ]


def _evaluation_fixture() -> tuple[list[RankedResult], list[HotpotQALabelRecord], list[MemoryGraph]]:
    predictions: list[RankedResult] = [
        {
            "task_id": "hotpot_eval_1",
            "method": "bm25",
            "ranked_nodes": [
                {"node_id": "m0", "score": 3.0},
                {"node_id": "m1", "score": 2.0},
                {"node_id": "m3", "score": 1.0},
                {"node_id": "m2", "score": 0.0},
            ],
            "retrieved_subgraph": {
                "nodes": ["m0", "m1", "m3"],
                "edges": [
                    {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 1.0, "directed": True},
                    {"source": "m0", "target": "m3", "edge_type": "bridge", "weight": 2.0, "directed": False},
                ],
            },
            "latency_ms": 7.0,
            "input_tokens": 5,
        }
    ]
    labels: list[HotpotQALabelRecord] = [
        {
            "task_id": "hotpot_eval_1",
            "gold_answer": "Paris",
            "gold_evidence_sentence_ids": ["m0", "m3"],
            "gold_dependency_edges": [],
        }
    ]
    graphs: list[MemoryGraph] = [
        {
            "task_id": "hotpot_eval_1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "Which city?"},
                {
                    "id": "m0",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "Evidence one.",
                    "source_ref": "A",
                    "group_key": "document:A",
                    "sequence_index": 0,
                    "metadata": {"title": "A", "position": 0},
                },
                {
                    "id": "m1",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "Distractor.",
                    "source_ref": "A",
                    "group_key": "document:A",
                    "sequence_index": 1,
                    "metadata": {"title": "A", "position": 1},
                },
                {
                    "id": "m2",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "Connector.",
                    "source_ref": "B",
                    "group_key": "document:B",
                    "sequence_index": 0,
                    "metadata": {"title": "B", "position": 2},
                },
                {
                    "id": "m3",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "Evidence two.",
                    "source_ref": "B",
                    "group_key": "document:B",
                    "sequence_index": 1,
                    "metadata": {"title": "B", "position": 3},
                },
            ],
            "edges": [
                {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 1.0, "directed": True},
                {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False},
                {"source": "m2", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False},
                {"source": "m0", "target": "m3", "edge_type": "bridge", "weight": 2.0, "directed": False},
            ],
        }
    ]
    return predictions, labels, graphs
