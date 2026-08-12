import pytest

from graph_memory.evaluation.connectivity import (
    connected_evidence_at,
    query_evidence_connectivity_at,
)
from graph_memory.evaluation.metrics import full_support_at
from graph_memory.evaluation.service import (
    evaluate_results,
)
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.results import RankedResult
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.tables import split_metric_tables


def _evidence_labels(labels: list[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=record.task_id,
            gold_answer=record.gold_answer,
            gold_evidence_item_ids=record.gold_evidence_sentence_ids,
            gold_dependency_edges=record.gold_dependency_edges,
        )
        for label in labels
        for record in (HotpotQALabelRecord.model_validate(label),)
    ]


def _graph(*edges: object) -> EvidenceGraph:
    return EvidenceGraph.model_validate({
        "task_id": "hotpot_ex1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "question"},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "zero",
            },
            {
                "id": "m1",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "one",
            },
            {
                "id": "m2",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "two",
            },
        ],
        "edges": list(edges),
    })


def test_full_support_and_connected_evidence_use_top_k_nodes_on_shared_graph():
    ranked = ["m0", "m2", "m1"]
    gold = {"m0", "m2"}
    graph = _graph(
        {
            "source": "m0",
            "target": "m2",
            "edge_type": "bridge",
            "weight": 1.0,
            "directed": False,
        }
    )

    assert full_support_at(ranked, gold, 2) == 1.0
    assert connected_evidence_at(ranked, gold, graph, 2) == 1.0


def test_query_evidence_connectivity_requires_reachability_from_question():
    ranked = ["m0", "m2", "m1"]
    gold = {"m0", "m2"}
    graph = _graph(
        {
            "source": "q",
            "target": "m0",
            "edge_type": "query_overlap",
            "weight": 1.0,
            "directed": True,
        },
        {
            "source": "m0",
            "target": "m2",
            "edge_type": "bridge",
            "weight": 1.0,
            "directed": False,
        },
    )

    assert query_evidence_connectivity_at(ranked, gold, graph, 10) == 1.0


def test_evaluate_results_joins_predictions_labels_and_graphs():
    predictions = [
        RankedResult.model_validate({
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
        })
    ]
    labels = [
        HotpotQALabelRecord.model_validate({
            "task_id": "hotpot_ex1",
            "gold_answer": "Paris",
            "gold_evidence_sentence_ids": ["m0", "m2"],
            "gold_dependency_edges": [],
        })
    ]
    graphs = [
        _graph(
            {
                "source": "q",
                "target": "m0",
                "edge_type": "query_overlap",
                "weight": 1.0,
                "directed": True,
            },
            {
                "source": "m0",
                "target": "m2",
                "edge_type": "bridge",
                "weight": 1.0,
                "directed": False,
            },
        )
    ]

    rows = evaluate_results(
        EvidenceEvaluationRequest(
            predictions=tuple(predictions),
            labels=tuple(_evidence_labels(labels)),
            graphs=tuple(graphs)
        )
    )

    assert [row.model_dump(mode="json", by_alias=True) for row in rows] == [
        {
            "Method": "bm25",
            "Evaluation Schema": "evidence_v3",
            "Recall@2": 1.0,
            "Recall@5": 1.0,
            "Recall@10": 1.0,
            "Evidence F1@5": 0.5714285714285715,
            "Evidence F1@10": 0.33333333333333337,
            "Evidence Density@5": "N/A",
            "Evidence Density@10": "N/A",
            "Coverage@256 Tokens": "N/A",
            "Coverage@512 Tokens": "N/A",
            "Coverage@1024 Tokens": "N/A",
            "Coverage@2048 Tokens": "N/A",
            "Coverage@4096 Tokens": "N/A",
            "Coverage@8192 Tokens": "N/A",
            "Full Support@256 Tokens": "N/A",
            "Full Support@512 Tokens": "N/A",
            "Full Support@1024 Tokens": "N/A",
            "Full Support@2048 Tokens": "N/A",
            "Full Support@4096 Tokens": "N/A",
            "Full Support@8192 Tokens": "N/A",
            "Coverage Budget-AUC": "N/A",
            "Full Support Budget-AUC": "N/A",
            "Span F1@2048 Tokens": "N/A",
            "Evidence Density@2048 Tokens": "N/A",
            "Full Support@5": 1.0,
            "Full Support@10": 1.0,
            "MRR": 1.0,
            "Connected Evidence Recall@5": 1.0,
            "Connected Evidence Recall@10": 1.0,
            "Query-Evidence Connectivity@10": 1.0,
            "Path Recall@10": "N/A",
            "Edge Recall@10": "N/A",
            "Edge Precision@10": "N/A",
            "Edge F1@10": "N/A",
            "Retrieval Latency / Query": 4.0,
            "Index Build Time": 0.0,
            "Graph Construction Time": 0.0,
            "Memory Size": 3.0,
            "Avg Retrieved Nodes": 2.0,
            "Avg Retrieved Edges": 0.0,
        }
    ]
    legacy = rows[0].model_copy(update={"evaluation_schema": "evidence_v2"})
    with pytest.raises(ValueError, match="mixed evaluation schemas"):
        split_metric_tables([rows[0], legacy])


def test_evaluate_results_rejects_task_id_mismatch():
    predictions = [
        RankedResult.model_validate({
            "task_id": "hotpot_ex1",
            "method": "bm25",
            "ranked_nodes": [],
            "retrieved_subgraph": {"nodes": [], "edges": []},
            "latency_ms": 0.0,
            "input_tokens": 0,
        })
    ]
    labels = [
        HotpotQALabelRecord.model_validate({
            "task_id": "hotpot_other",
            "gold_answer": "unknown",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        })
    ]
    graphs = [_graph()]

    with pytest.raises(ValueError, match="must align"):
        evaluate_results(
            EvidenceEvaluationRequest(
                predictions=tuple(predictions),
            labels=tuple(_evidence_labels(labels)),
            graphs=tuple(graphs)
            )
        )
