from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.connectivity import (
    GraphConnectivity,
    connected_evidence_at,
    query_evidence_connectivity_at,
)
from graph_memory.evaluation.failure_cases import build_failure_cases
from graph_memory.evaluation.metrics import evidence_f1_at, full_support_at, mrr, recall_at
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


def test_evaluation_domain_modules_own_public_evaluation_logic() -> None:
    assert recall_at.__module__ == "graph_memory.evaluation.metrics"
    assert evidence_f1_at.__module__ == "graph_memory.evaluation.metrics"
    assert connected_evidence_at.__module__ == "graph_memory.evaluation.connectivity"
    assert GraphConnectivity.__module__ == "graph_memory.evaluation.connectivity"
    assert evaluate_results.__module__ == "graph_memory.evaluation.service"
    assert split_metric_tables.__module__ == "graph_memory.evaluation.tables"
    assert build_failure_cases.__module__ == "graph_memory.evaluation.failure_cases"


def test_metric_connectivity_table_and_failure_case_outputs_stay_stable() -> None:
    predictions, labels, graphs = _evaluation_fixture()
    graph = graphs[0]
    ranked_node_ids = [record["node_id"] for record in predictions[0]["ranked_nodes"]]
    gold_nodes = set(labels[0]["gold_evidence_sentence_ids"])

    connectivity = GraphConnectivity.from_graph(graph, allowed_nodes={"q", "m0", "m1", "m2"})

    assert recall_at(ranked_node_ids, gold_nodes, 2) == 0.5
    assert evidence_f1_at(ranked_node_ids, gold_nodes, 2) == 0.5
    assert full_support_at(ranked_node_ids, gold_nodes, 3) == 1.0
    assert mrr(ranked_node_ids, gold_nodes) == 1.0
    assert connectivity.undirected_reachable("m0") == {"q", "m0", "m1", "m2"}
    assert connectivity.directed_reachable("q") == {"q", "m0", "m1", "m2"}
    assert connected_evidence_at(ranked_node_ids, gold_nodes, graph, 3) == 1.0
    assert query_evidence_connectivity_at(ranked_node_ids, gold_nodes, graph, 10) == 1.0

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

    main_rows, path_rows, efficiency_rows = split_metric_tables(rows)

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

    failure_prediction = cast(
        RankedResult,
        cast(
            object,
        {
            **predictions[0],
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


def test_scripts_and_tests_import_evaluation_domain_modules_directly() -> None:
    forbidden_importers: list[tuple[str, int]] = []
    for path in _python_files(("graph_memory", "scripts", "tests")):
        if path == Path("graph_memory/evaluation/__init__.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "graph_memory.evaluation":
                forbidden_importers.append((str(path), node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "graph_memory.evaluation":
                        forbidden_importers.append((str(path), node.lineno))

    assert forbidden_importers == []


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


def _python_files(roots: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in Path(root).rglob("*.py") if ".pytest_tmp" not in path.parts)
    return files
