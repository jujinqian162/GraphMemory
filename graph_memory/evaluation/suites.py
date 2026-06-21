from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import FailureCase, MetricRow, MetricTableRow, TaskMetricRow
from graph_memory.evaluation.connectivity import connected_evidence_at, query_evidence_connectivity_at
from graph_memory.evaluation.metrics import evidence_f1_at, full_support_at, mrr, recall_at
from graph_memory.evaluation.path_metrics import edge_recall_at, path_recall_at
from graph_memory.evaluation.requests import EvidenceEvaluationRequest
from graph_memory.registry.methods import build_method_registry
from graph_memory.validation.common import ContractValidationError, validate_task_id_alignment
from graph_memory.validation.metrics import validate_evidence_metric_rows

EVIDENCE_METRIC_COLUMNS = [
    "Method",
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
    "Path Recall@10",
    "Edge Recall@10",
    "Retrieval Latency / Query",
]
METHOD_REGISTRY = build_method_registry()


class MetricSuite(Protocol):
    name: str

    def evaluate(self, request: object) -> list[MetricTableRow]:
        ...

    def validate_metric_rows(self, rows: object) -> None:
        ...

    def build_failure_cases(self, request: object, *, top_k: int = 10, limit: int = 0) -> list[dict[str, object]]:
        ...


@dataclass(frozen=True)
class EvidenceMetricSuite:
    name: str = "evidence"

    def evaluate(self, request: EvidenceEvaluationRequest) -> list[MetricRow]:
        prediction_task_ids = {prediction["task_id"] for prediction in request.predictions}
        label_task_ids = {label.task_id for label in request.labels}
        graph_task_ids = {graph["task_id"] for graph in request.graphs}
        validate_task_id_alignment("prediction/label join", prediction_task_ids, label_task_ids)
        validate_task_id_alignment("prediction/graph join", prediction_task_ids, graph_task_ids)

        labels_by_task_id = {label.task_id: label for label in request.labels}
        graphs_by_task_id = {graph["task_id"]: graph for graph in request.graphs}
        methods = {prediction["method"] for prediction in request.predictions}
        if len(methods) != 1:
            raise ContractValidationError(
                f"Invalid evaluation input: expected one method per file, got methods={sorted(methods)}."
            )
        method = next(iter(methods)) if methods else ""
        path_metrics_supported = _path_metrics_supported(method)
        path_recall_values: list[float] = []
        edge_recall_values: list[float] = []

        per_task_rows: list[TaskMetricRow] = []
        for prediction in request.predictions:
            task_id = prediction["task_id"]
            ranked_node_ids = [ranked_node["node_id"] for ranked_node in prediction["ranked_nodes"]]
            label = labels_by_task_id[task_id]
            graph = graphs_by_task_id[task_id]
            gold_nodes = set(label.gold_evidence_item_ids)
            _validate_gold_nodes_exist(task_id, gold_nodes, graph)
            if path_metrics_supported and label.gold_dependency_edges:
                gold_dependency_edges = set(label.gold_dependency_edges)
                path_recall_values.append(path_recall_at(prediction["retrieved_subgraph"], gold_dependency_edges))
                edge_recall_values.append(edge_recall_at(prediction["retrieved_subgraph"], gold_dependency_edges))
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
                    "Query-Evidence Connectivity@10": query_evidence_connectivity_at(
                        ranked_node_ids,
                        gold_nodes,
                        graph,
                        10,
                    ),
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
        aggregate_row["Path Recall@10"] = _mean_optional(path_recall_values)
        aggregate_row["Edge Recall@10"] = _mean_optional(edge_recall_values)
        return [aggregate_row]

    def validate_metric_rows(self, rows: object) -> None:
        validate_evidence_metric_rows(rows)

    def build_failure_cases(
        self,
        request: EvidenceEvaluationRequest,
        *,
        top_k: int = 10,
        limit: int = 0,
    ) -> list[FailureCase]:
        if limit <= 0:
            return []
        labels_by_task_id = {label.task_id: label for label in request.labels}
        graphs_by_task_id = {graph["task_id"]: graph for graph in request.graphs}
        cases: list[FailureCase] = []
        for prediction in request.predictions:
            task_id = prediction["task_id"]
            ranked_node_ids = [ranked_node["node_id"] for ranked_node in prediction["ranked_nodes"]]
            gold_nodes = set(labels_by_task_id[task_id].gold_evidence_item_ids)
            if full_support_at(ranked_node_ids, gold_nodes, top_k) == 1.0:
                continue
            retrieved_top_k = ranked_node_ids[:top_k]
            cases.append(
                {
                    "debug_type": "failure_case",
                    "task_id": task_id,
                    "method": prediction["method"],
                    "failure_type": f"missing_full_support_at_{top_k}",
                    "gold_evidence_item_ids": sorted(gold_nodes),
                    "retrieved_top_k": retrieved_top_k,
                    "missing_gold_nodes": sorted(gold_nodes - set(retrieved_top_k)),
                    "connected_gold_in_top_k": bool(
                        connected_evidence_at(ranked_node_ids, gold_nodes, graphs_by_task_id[task_id], top_k)
                    ),
                }
            )
            if len(cases) >= limit:
                break
        return cases


def evidence_metric_suite() -> EvidenceMetricSuite:
    return EvidenceMetricSuite()


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


def _path_metrics_supported(method: str) -> bool:
    try:
        return METHOD_REGISTRY.supports_path_metrics(method)
    except ValueError:
        return False


def _mean_optional(values: Iterable[float]) -> float | str:
    materialized = list(values)
    if not materialized:
        return "N/A"
    return sum(materialized) / len(materialized)

__all__ = ["EVIDENCE_METRIC_COLUMNS", "EvidenceMetricSuite", "MetricSuite", "evidence_metric_suite"]
