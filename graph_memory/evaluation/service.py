from __future__ import annotations

from collections.abc import Iterable

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow, TaskMetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskLabels
from graph_memory.evaluation.connectivity import connected_evidence_at, query_evidence_connectivity_at
from graph_memory.evaluation.metrics import evidence_f1_at, full_support_at, mrr, recall_at
from graph_memory.validation import ContractValidationError, validate_task_id_alignment


def evaluate_results(
    predictions: list[RankedResult], labels: list[MemoryTaskLabels], graphs: list[MemoryGraph]
) -> list[MetricRow]:
    prediction_task_ids = {prediction["task_id"] for prediction in predictions}
    label_task_ids = {label["task_id"] for label in labels}
    graph_task_ids = {graph["task_id"] for graph in graphs}
    validate_task_id_alignment("prediction/label join", prediction_task_ids, label_task_ids)
    validate_task_id_alignment("prediction/graph join", prediction_task_ids, graph_task_ids)

    labels_by_task_id = {label["task_id"]: label for label in labels}
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    methods = {prediction["method"] for prediction in predictions}
    if len(methods) != 1:
        raise ContractValidationError(f"Invalid evaluation input: expected one method per file, got methods={sorted(methods)}.")
    method = next(iter(methods)) if methods else ""

    per_task_rows: list[TaskMetricRow] = []
    for prediction in predictions:
        task_id = prediction["task_id"]
        ranked_node_ids = [ranked_node["node_id"] for ranked_node in prediction["ranked_nodes"]]
        label = labels_by_task_id[task_id]
        graph = graphs_by_task_id[task_id]
        gold_nodes = set(label["gold_evidence_nodes"])
        _validate_gold_nodes_exist(task_id, gold_nodes, graph)
        per_task_rows.append(
            {
                "Recall@2": recall_at(ranked_node_ids, gold_nodes, 2),
                "Recall@5": recall_at(ranked_node_ids, gold_nodes, 5),
                "Recall@10": recall_at(ranked_node_ids, gold_nodes, 10),
                "Evidence F1@5": evidence_f1_at(ranked_node_ids, gold_nodes, 5),
                "Evidence F1@10": evidence_f1_at(ranked_node_ids, gold_nodes, 10),
                "Full Support@5": full_support_at(ranked_node_ids, gold_nodes, 5),
                "Full Support@10": full_support_at(ranked_node_ids, gold_nodes, 10),
                "MRR": mrr(ranked_node_ids, gold_nodes),
                "Connected Evidence Recall@5": connected_evidence_at(ranked_node_ids, gold_nodes, graph, 5),
                "Connected Evidence Recall@10": connected_evidence_at(ranked_node_ids, gold_nodes, graph, 10),
                "Query-Evidence Connectivity@10": query_evidence_connectivity_at(ranked_node_ids, gold_nodes, graph, 10),
                "Retrieval Latency / Query": float(prediction["latency_ms"]),
                "Memory Size": float(_memory_node_count(graph)),
                "Avg Retrieved Nodes": float(len(prediction["retrieved_subgraph"]["nodes"])),
                "Avg Retrieved Edges": float(len(prediction["retrieved_subgraph"]["edges"])),
            }
        )

    aggregate_row: MetricRow = {
        "Method": method,
        "Recall@2": 0.0,
        "Recall@5": 0.0,
        "Recall@10": 0.0,
        "Evidence F1@5": 0.0,
        "Evidence F1@10": 0.0,
        "Full Support@5": 0.0,
        "Full Support@10": 0.0,
        "MRR": 0.0,
        "Connected Evidence Recall@5": 0.0,
        "Connected Evidence Recall@10": 0.0,
        "Query-Evidence Connectivity@10": 0.0,
        "Path Recall@10": "N/A",
        "Edge Recall@10": "N/A",
        "Retrieval Latency / Query": 0.0,
        "Index Build Time": 0.0,
        "Graph Construction Time": 0.0,
        "Memory Size": 0.0,
        "Avg Retrieved Nodes": 0.0,
        "Avg Retrieved Edges": 0.0,
    }
    for column in [
        "Recall@2",
        "Recall@5",
        "Recall@10",
        "Evidence F1@5",
        "Evidence F1@10",
        "Full Support@5",
        "Full Support@10",
        "MRR",
        "Connected Evidence Recall@5",
        "Connected Evidence Recall@10",
        "Query-Evidence Connectivity@10",
        "Retrieval Latency / Query",
        "Memory Size",
        "Avg Retrieved Nodes",
        "Avg Retrieved Edges",
    ]:
        aggregate_row[column] = _mean(row[column] for row in per_task_rows)
    return [aggregate_row]


def _validate_gold_nodes_exist(task_id: str, gold_nodes: set[NodeId], graph: MemoryGraph) -> None:
    graph_node_ids = {str(node.get("id")) for node in graph.get("nodes", [])}
    missing = sorted(gold_nodes - graph_node_ids)
    if missing:
        raise ContractValidationError(f"Invalid evaluation input: task_id={task_id} gold nodes missing from graph: {missing}.")


def _memory_node_count(graph: MemoryGraph) -> int:
    return sum(1 for node in graph.get("nodes", []) if node.get("id") != "q")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


__all__ = ["evaluate_results"]
