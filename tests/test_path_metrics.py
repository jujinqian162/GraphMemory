from __future__ import annotations

from graph_memory.contracts.graphs import GraphEdge, MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.evaluation.path_metrics import edge_recall_at, path_recall_at
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.service import evaluate_results


def _subgraph(edges: list[GraphEdge]) -> dict[str, object]:
    return {"nodes": ["m0", "m1", "m2"], "edges": edges}


def test_edge_recall_requires_direct_visible_edge_and_respects_direction() -> None:
    gold_edges = {("m0", "m2"), ("m2", "m1")}
    subgraph = _subgraph(
        [
            {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
            {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
        ]
    )

    assert edge_recall_at(subgraph, gold_edges) == 0.5


def test_edge_recall_treats_undirected_edges_as_covering_either_direction() -> None:
    subgraph = _subgraph(
        [{"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False}]
    )

    assert edge_recall_at(subgraph, {("m2", "m1")}) == 1.0


def test_path_recall_allows_multihop_visible_paths() -> None:
    subgraph = _subgraph(
        [
            {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": True},
            {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
        ]
    )

    assert path_recall_at(subgraph, {("m0", "m2")}) == 1.0
    assert path_recall_at(subgraph, {("m2", "m0")}) == 0.0


def test_evaluation_keeps_flat_methods_path_metrics_na_even_with_gold_edges() -> None:
    rows = evaluate_results(_request(method="bm25", retrieved_edges=[]))

    assert rows[0]["Path Recall@10"] == "N/A"
    assert rows[0]["Edge Recall@10"] == "N/A"


def test_evaluation_keeps_dense_ft_path_metrics_na_for_twowiki_gold_edges() -> None:
    rows = evaluate_results(_request(method="dense_ft", retrieved_edges=[]))

    assert rows[0]["Path Recall@10"] == "N/A"
    assert rows[0]["Edge Recall@10"] == "N/A"


def test_evaluation_uses_registry_capability_not_prediction_metadata_override() -> None:
    rows = evaluate_results(
        _request(
            method="bm25",
            retrieved_edges=[{"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False}],
            metadata={"path_metrics_supported": True},
        )
    )

    assert rows[0]["Path Recall@10"] == "N/A"
    assert rows[0]["Edge Recall@10"] == "N/A"


def test_evaluation_computes_path_metrics_for_graph_aware_methods() -> None:
    rows = evaluate_results(
        _request(
            method="dense_graph_rerank",
            retrieved_edges=[
                {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": True},
                {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
            ],
        )
    )

    assert rows[0]["Path Recall@10"] == 1.0
    assert rows[0]["Edge Recall@10"] == 0.0


def test_evaluation_computes_numeric_path_metrics_for_rgcn_with_twowiki_gold_edges() -> None:
    rows = evaluate_results(
        _request(
            method="dense_rgcn_graph_retriever",
            retrieved_edges=[
                {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": True},
                {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
            ],
        )
    )

    assert isinstance(rows[0]["Path Recall@10"], float)
    assert isinstance(rows[0]["Edge Recall@10"], float)


def test_evaluation_counts_rgcn_missing_twowiki_path_as_zero() -> None:
    rows = evaluate_results(
        _request(
            method="dense_rgcn_graph_retriever",
            retrieved_edges=[
                {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
            ],
        )
    )

    assert rows[0]["Path Recall@10"] == 0.0
    assert rows[0]["Edge Recall@10"] == 0.0


def _request(
    *,
    method: str,
    retrieved_edges: list[GraphEdge],
    metadata: dict[str, object] | None = None,
) -> EvidenceEvaluationRequest:
    prediction: RankedResult = {
        "task_id": "2wiki_abc123",
        "method": method,
        "ranked_nodes": [
            {"node_id": "m0", "score": 3.0},
            {"node_id": "m1", "score": 2.0},
            {"node_id": "m2", "score": 1.0},
        ],
        "retrieved_subgraph": {"nodes": ["m0", "m1", "m2"], "edges": retrieved_edges},
        "latency_ms": 2.0,
        "input_tokens": 4,
    }
    if metadata is not None:
        prediction["metadata"] = metadata
    label = EvidenceLabel(
        task_id="2wiki_abc123",
        gold_answer="Beth",
        gold_evidence_item_ids=("m0", "m2"),
        gold_dependency_edges=(("m0", "m2"),),
    )
    graph: MemoryGraph = {
        "task_id": "2wiki_abc123",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Who?"},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": "A"},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": "B"},
            {"id": "m2", "node_type": "graph_item", "node_kind": "document_sentence", "text": "C"},
        ],
        "edges": [
            {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": True},
            {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": True},
        ],
    }
    return EvidenceEvaluationRequest(predictions=[prediction], labels=[label], graphs=[graph])
