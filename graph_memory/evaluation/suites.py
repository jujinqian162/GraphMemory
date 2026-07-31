from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from graph_memory.contracts.common import NodeId
from graph_memory.evaluation.connectivity import (
    connected_evidence_at,
    query_evidence_connectivity_at,
)
from graph_memory.evaluation.contracts import (
    EVIDENCE_METRIC_COLUMNS,
    FailureCase,
    MetricRow,
    PerTaskMetricRow,
    TaskMetricRow,
)
from graph_memory.evaluation.metrics import (
    evidence_f1_at,
    full_support_at,
    mrr,
    recall_at,
)
from graph_memory.evaluation.path_metrics import path_recall_at
from graph_memory.evaluation.requests import EvidenceEvaluationRequest
from graph_memory.graphs.contracts import EvidenceGraph

class MetricSuite(Protocol):
    name: str

    def evaluate(self, request: object) -> list[MetricRow]: ...

    def build_failure_cases(
        self, request: object, *, top_k: int = 10, limit: int = 0
    ) -> list[FailureCase]: ...


@dataclass(frozen=True)
class EvidenceMetricSuite:
    name: str = "evidence"

    def evaluate(self, request: EvidenceEvaluationRequest) -> list[MetricRow]:
        aggregate_rows, _ = self._evaluate_impl(request)
        return aggregate_rows

    def evaluate_with_per_task(
        self, request: EvidenceEvaluationRequest
    ) -> tuple[list[MetricRow], list[PerTaskMetricRow]]:
        return self._evaluate_impl(request)

    def _evaluate_impl(
        self, request: EvidenceEvaluationRequest
    ) -> tuple[list[MetricRow], list[PerTaskMetricRow]]:
        if not request.predictions:
            raise ValueError("evaluation requires at least one prediction")
        labels_by_task_id = {label.task_id: label for label in request.labels}
        graphs_by_task_id = {graph.task_id: graph for graph in request.graphs}
        methods = {prediction.method for prediction in request.predictions}
        if len(methods) != 1:
            raise ValueError(
                f"expected one method per evaluation input, got={sorted(methods)}"
            )
        method = next(iter(methods))
        path_metrics_supported = any(
            label.gold_dependency_edges for label in request.labels
        )
        query_connectivity_supported = _query_connectivity_supported(request.graphs)
        path_recall_values: list[float] = []
        edge_true_positive_count = 0
        predicted_edge_count = 0
        gold_edge_count = 0

        task_rows: list[TaskMetricRow] = []
        per_task_rows: list[PerTaskMetricRow] = []
        for prediction in request.predictions:
            task_id = prediction.task_id
            ranked_node_ids = list(prediction.ranked_node_ids)
            label = labels_by_task_id[task_id]
            graph = graphs_by_task_id.get(task_id)
            gold_nodes = set(label.gold_evidence_item_ids)
            if graph is not None:
                _validate_gold_nodes_exist(task_id, gold_nodes, graph)
            if path_metrics_supported and label.gold_dependency_edges:
                gold_dependency_edges = set(label.gold_dependency_edges)
                top_ten = set(ranked_node_ids[:10])
                predicted_edges = {
                    (edge.source, edge.target)
                    for edge in prediction.retrieved_subgraph.edges
                    if edge.source in top_ten and edge.target in top_ten
                }
                edge_true_positive_count += len(
                    predicted_edges & gold_dependency_edges
                )
                predicted_edge_count += len(predicted_edges)
                gold_edge_count += len(gold_dependency_edges)
                path_recall_values.append(
                    path_recall_at(
                        prediction.retrieved_subgraph, gold_dependency_edges
                    )
                )
            task_row = TaskMetricRow.model_validate(
                {
                    "Recall@2": recall_at(ranked_node_ids, gold_nodes, 2),
                    "Recall@5": recall_at(ranked_node_ids, gold_nodes, 5),
                    "Recall@10": recall_at(ranked_node_ids, gold_nodes, 10),
                    "Evidence F1@5": evidence_f1_at(
                        ranked_node_ids, gold_nodes, 5
                    ),
                    "Evidence F1@10": evidence_f1_at(
                        ranked_node_ids, gold_nodes, 10
                    ),
                    "Evidence Density@5": "N/A",
                    "Evidence Density@10": "N/A",
                    "Full Support@5": full_support_at(
                        ranked_node_ids, gold_nodes, 5
                    ),
                    "Full Support@10": full_support_at(
                        ranked_node_ids, gold_nodes, 10
                    ),
                    "MRR": mrr(ranked_node_ids, gold_nodes),
                    "Connected Evidence Recall@5": (
                        connected_evidence_at(ranked_node_ids, gold_nodes, graph, 5)
                        if graph is not None
                        else 0.0
                    ),
                    "Connected Evidence Recall@10": (
                        connected_evidence_at(
                            ranked_node_ids, gold_nodes, graph, 10
                        )
                        if graph is not None
                        else 0.0
                    ),
                    "Query-Evidence Connectivity@10": (
                        query_evidence_connectivity_at(
                            ranked_node_ids, gold_nodes, graph, 10
                        )
                        if graph is not None
                        else 0.0
                    ),
                    "Retrieval Latency / Query": prediction.latency_ms,
                    "Memory Size": (
                        float(_memory_node_count(graph)) if graph is not None else 0.0
                    ),
                    "Avg Retrieved Nodes": float(
                        len(prediction.retrieved_subgraph.nodes)
                    ),
                    "Avg Retrieved Edges": float(
                        len(prediction.retrieved_subgraph.edges)
                    ),
                }
            )
            task_rows.append(task_row)
            per_task_rows.append(
                PerTaskMetricRow(
                    task_id=task_id,
                    query_intent=label.query_intent,
                    motif_type=label.motif_type,
                    review_status=label.review_status,
                    **task_row.model_dump(mode="python"),
                )
            )

        aggregate: dict[str, object] = {
            "Method": method,
            "Evaluation Schema": _evaluation_schema(request.graphs),
            "Path Recall@10": _mean_optional(path_recall_values),
            "Edge Recall@10": "N/A",
            "Edge Precision@10": "N/A",
            "Edge F1@10": "N/A",
            "Index Build Time": 0.0,
            "Graph Construction Time": 0.0,
        }
        task_alias_rows = [
            row.model_dump(mode="python", by_alias=True) for row in task_rows
        ]
        for column in TaskMetricRow.model_fields:
            alias = TaskMetricRow.model_fields[column].serialization_alias or column
            values = [row[alias] for row in task_alias_rows]
            aggregate[alias] = (
                "N/A"
                if any(value == "N/A" for value in values)
                else _mean(float(value) for value in values)
            )
        if path_metrics_supported:
            edge_precision = (
                edge_true_positive_count / predicted_edge_count
                if predicted_edge_count
                else 0.0
            )
            edge_recall = (
                edge_true_positive_count / gold_edge_count if gold_edge_count else 0.0
            )
            aggregate["Edge Precision@10"] = edge_precision
            aggregate["Edge Recall@10"] = edge_recall
            aggregate["Edge F1@10"] = (
                2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)
                if edge_precision + edge_recall
                else 0.0
            )
        if not request.graphs:
            aggregate["Connected Evidence Recall@5"] = "N/A"
            aggregate["Connected Evidence Recall@10"] = "N/A"
            aggregate["Query-Evidence Connectivity@10"] = "N/A"
            aggregate["Memory Size"] = "N/A"
        elif not query_connectivity_supported:
            aggregate["Query-Evidence Connectivity@10"] = "N/A"
        return [MetricRow.model_validate(aggregate)], per_task_rows

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
        graphs_by_task_id = {graph.task_id: graph for graph in request.graphs}
        cases: list[FailureCase] = []
        for prediction in request.predictions:
            task_id = prediction.task_id
            ranked_node_ids = list(prediction.ranked_node_ids)
            gold_nodes = set(labels_by_task_id[task_id].gold_evidence_item_ids)
            if full_support_at(ranked_node_ids, gold_nodes, top_k) == 1.0:
                continue
            retrieved_top_k = tuple(ranked_node_ids[:top_k])
            cases.append(
                FailureCase(
                    debug_type="failure_case",
                    task_id=task_id,
                    method=prediction.method,
                    failure_type=f"missing_full_support_at_{top_k}",
                    gold_evidence_item_ids=tuple(sorted(gold_nodes)),
                    retrieved_top_k=retrieved_top_k,
                    missing_gold_nodes=tuple(
                        sorted(gold_nodes - set(retrieved_top_k))
                    ),
                    connected_gold_in_top_k=(
                        bool(
                            connected_evidence_at(
                                ranked_node_ids,
                                gold_nodes,
                                graphs_by_task_id[task_id],
                                top_k,
                            )
                        )
                        if task_id in graphs_by_task_id
                        else False
                    ),
                )
            )
            if len(cases) >= limit:
                break
        return cases


def evidence_metric_suite() -> EvidenceMetricSuite:
    return EvidenceMetricSuite()


def _validate_gold_nodes_exist(
    task_id: str, gold_nodes: set[NodeId], graph: EvidenceGraph
) -> None:
    missing = sorted(gold_nodes - graph.graph_item_ids)
    if missing:
        raise ValueError(
            f"task_id={task_id} gold nodes missing from graph: {missing}"
        )


def _evaluation_schema(graphs: tuple[EvidenceGraph, ...]) -> str:
    schemas = {
        value
        for graph in graphs
        if isinstance(
            (value := (graph.metadata or {}).get("evaluation_schema")), str
        )
    }
    if not schemas:
        return "evidence_v3"
    if len(schemas) != 1:
        raise ValueError(f"mixed graph evaluation schemas={sorted(schemas)}")
    return next(iter(schemas))


def _query_connectivity_supported(graphs: tuple[EvidenceGraph, ...]) -> bool:
    return all(
        (graph.metadata or {}).get("query_edges_supported", True) is not False
        for graph in graphs
    )


def _memory_node_count(graph: EvidenceGraph) -> int:
    return len(graph.graph_item_ids)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _mean_optional(values: Iterable[float]) -> float | str:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else "N/A"


__all__ = [
    "EVIDENCE_METRIC_COLUMNS",
    "EvidenceMetricSuite",
    "MetricSuite",
    "evidence_metric_suite",
]
