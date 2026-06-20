import pytest
from typing import cast

from graph_memory.evaluation.connectivity import (
    connected_evidence_at,
    query_evidence_connectivity_at,
)
from graph_memory.evaluation.metrics import (
    evidence_f1_at,
    full_support_at,
    mrr,
    recall_at,
)
from graph_memory.evaluation.service import (
    evaluate_results,
)
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.validation import ContractValidationError


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


def test_node_metrics_use_ranked_nodes_and_gold_nodes():
    ranked = ["m2", "m0", "m1"]
    gold = {"m0", "m1"}

    assert recall_at(ranked, gold, 2) == 0.5
    assert evidence_f1_at(ranked, gold, 2) == 0.5
    assert full_support_at(ranked, gold, 2) == 0.0
    assert full_support_at(ranked, gold, 3) == 1.0
    assert mrr(ranked, gold) == 0.5


def test_full_support_and_connected_evidence_use_top_k_nodes_on_shared_graph():
    ranked = ["m0", "m2", "m1"]
    gold = {"m0", "m2"}
    graph = cast(
        MemoryGraph,
        cast(object, {"task_id": "hotpot_ex1", "nodes": [], "edges": [{"source": "m0", "target": "m2", "edge_type": "bridge"}]}),
    )

    assert full_support_at(ranked, gold, 2) == 1.0
    assert connected_evidence_at(ranked, gold, graph, 2) == 1.0


def test_query_evidence_connectivity_requires_reachability_from_question():
    ranked = ["m0", "m2", "m1"]
    gold = {"m0", "m2"}
    graph = cast(
        MemoryGraph,
        cast(
            object,
            {
                "task_id": "hotpot_ex1",
                "nodes": [],
                "edges": [
                    {"source": "q", "target": "m0", "edge_type": "query_overlap", "directed": True},
                    {"source": "m0", "target": "m2", "edge_type": "bridge", "directed": False},
                ],
            },
        ),
    )

    assert query_evidence_connectivity_at(ranked, gold, graph, 10) == 1.0


def test_evaluate_results_joins_predictions_labels_and_graphs():
    predictions: list[RankedResult] = [
        {
            "task_id": "hotpot_ex1",
            "method": "bm25",
            "ranked_nodes": [
                {"node_id": "m0", "score": 2.0},
                {"node_id": "m2", "score": 1.0},
                {"node_id": "m1", "score": 0.0},
            ],
            "retrieved_subgraph": {"nodes": ["m0", "m2"], "edges": []},
            "latency_ms": 4.0,
            "input_tokens": 10,
        }
    ]
    labels: list[HotpotQALabelRecord] = [
        {
            "task_id": "hotpot_ex1",
            "gold_answer": "Paris",
            "gold_evidence_sentence_ids": ["m0", "m2"],
            "gold_dependency_edges": [],
        }
    ]
    graphs = cast(
        list[MemoryGraph],
        cast(
            object,
            [
                {
                    "task_id": "hotpot_ex1",
                    "nodes": [{"id": "q"}, {"id": "m0"}, {"id": "m1"}, {"id": "m2"}],
                    "edges": [
                        {"source": "q", "target": "m0", "edge_type": "query_overlap", "directed": True},
                        {"source": "m0", "target": "m2", "edge_type": "bridge", "directed": False},
                    ],
                }
            ],
        ),
    )

    rows = evaluate_results(EvidenceEvaluationRequest(predictions=predictions, labels=_evidence_labels(labels), graphs=graphs))

    assert rows == [
        {
            "Method": "bm25",
            "Recall@2": 1.0,
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
            "Retrieval Latency / Query": 4.0,
            "Index Build Time": 0.0,
            "Graph Construction Time": 0.0,
            "Memory Size": 3.0,
            "Avg Retrieved Nodes": 2.0,
            "Avg Retrieved Edges": 0.0,
        }
    ]


def test_evaluate_results_rejects_task_id_mismatch():
    predictions: list[RankedResult] = [{"task_id": "hotpot_ex1", "method": "bm25", "ranked_nodes": [], "retrieved_subgraph": {"nodes": [], "edges": []}, "latency_ms": 0.0, "input_tokens": 0}]
    labels: list[HotpotQALabelRecord] = [{"task_id": "hotpot_other", "gold_answer": "", "gold_evidence_sentence_ids": ["m0"], "gold_dependency_edges": []}]
    graphs: list[MemoryGraph] = [{"task_id": "hotpot_ex1", "nodes": [], "edges": []}]

    with pytest.raises(ContractValidationError, match="task_id"):
        evaluate_results(EvidenceEvaluationRequest(predictions=predictions, labels=_evidence_labels(labels), graphs=graphs))
