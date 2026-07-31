from __future__ import annotations

from dataclasses import dataclass

from graph_memory.evaluation.contracts import (
    FailureCase,
    MetricRow,
    PerTaskMetricRow,
    TaskMetricRow,
)
from graph_memory.evaluation.requests import SpanEvidenceEvaluationRequest
from graph_memory.evaluation.span_metrics import (
    connected_span_coverage_at,
    missing_gold_spans,
    span_dependency_edge_counts_at,
    span_dependency_path_recall_at,
    span_metrics_at,
    span_metrics_under_token_budget,
    span_mrr,
)

CONTEXT_TOKEN_BUDGET = 2048


@dataclass(frozen=True)
class SpanEvidenceMetricSuite:
    name: str = "span_evidence"

    def evaluate_with_per_task(
        self,
        request: SpanEvidenceEvaluationRequest,
    ) -> tuple[list[MetricRow], list[PerTaskMetricRow]]:
        if not request.predictions:
            raise ValueError("evaluation requires at least one prediction")
        methods = {prediction.method for prediction in request.predictions}
        if len(methods) != 1:
            raise ValueError("span evaluation requires exactly one retrieval method")
        method = next(iter(methods))
        labels = {label.task_id: label for label in request.labels}
        task_rows: list[TaskMetricRow] = []
        per_task: list[PerTaskMetricRow] = []
        path_recall_values: list[float] = []
        matched_predicted_edges = 0
        predicted_edges = 0
        matched_gold_edges = 0
        gold_edges = 0
        for prediction in request.predictions:
            label = labels[prediction.task_id]
            at_2 = span_metrics_at(
                prediction.ranked_nodes, label.gold_evidence_spans, 2
            )
            at_5 = span_metrics_at(
                prediction.ranked_nodes, label.gold_evidence_spans, 5
            )
            at_10 = span_metrics_at(
                prediction.ranked_nodes, label.gold_evidence_spans, 10
            )
            at_budget = span_metrics_under_token_budget(
                prediction.ranked_nodes,
                label.gold_evidence_spans,
                CONTEXT_TOKEN_BUDGET,
            )
            if label.gold_dependency_edges:
                path_recall_values.append(
                    span_dependency_path_recall_at(
                        prediction.ranked_nodes,
                        label.gold_dependency_edges,
                        prediction.retrieved_subgraph,
                        10,
                    )
                )
                edge_counts = span_dependency_edge_counts_at(
                    prediction.ranked_nodes,
                    label.gold_dependency_edges,
                    prediction.retrieved_subgraph,
                    10,
                )
                matched_predicted_edges += edge_counts.matched_predictions
                predicted_edges += edge_counts.predicted
                matched_gold_edges += edge_counts.matched_gold
                gold_edges += edge_counts.gold
            row = TaskMetricRow.model_validate(
                {
                    "Recall@2": at_2.coverage,
                    "Recall@5": at_5.coverage,
                    "Recall@10": at_10.coverage,
                    "Evidence F1@5": at_5.f1,
                    "Evidence F1@10": at_10.f1,
                    "Evidence Density@5": at_5.density,
                    "Evidence Density@10": at_10.density,
                    "Coverage@2048 Tokens": at_budget.coverage,
                    "Span F1@2048 Tokens": at_budget.f1,
                    "Evidence Density@2048 Tokens": at_budget.density,
                    "Full Support@2048 Tokens": at_budget.full_support,
                    "Full Support@5": at_5.full_support,
                    "Full Support@10": at_10.full_support,
                    "MRR": span_mrr(
                        prediction.ranked_nodes, label.gold_evidence_spans
                    ),
                    "Connected Evidence Recall@5": connected_span_coverage_at(
                        prediction.ranked_nodes,
                        label.gold_evidence_spans,
                        prediction.retrieved_subgraph,
                        5,
                    ),
                    "Connected Evidence Recall@10": connected_span_coverage_at(
                        prediction.ranked_nodes,
                        label.gold_evidence_spans,
                        prediction.retrieved_subgraph,
                        10,
                    ),
                    "Query-Evidence Connectivity@10": 0.0,
                    "Retrieval Latency / Query": prediction.latency_ms,
                    "Memory Size": float(len(prediction.ranked_nodes)),
                    "Avg Retrieved Nodes": float(
                        len(prediction.retrieved_subgraph.nodes)
                    ),
                    "Avg Retrieved Edges": float(
                        len(prediction.retrieved_subgraph.edges)
                    ),
                }
            )
            task_rows.append(row)
            per_task.append(
                PerTaskMetricRow(
                    task_id=prediction.task_id,
                    query_intent=label.query_intent,
                    motif_type=label.motif_type,
                    review_status=label.review_status,
                    **row.model_dump(mode="python"),
                )
            )
        edge_precision = (
            matched_predicted_edges / predicted_edges if predicted_edges else 0.0
        )
        edge_recall = matched_gold_edges / gold_edges if gold_edges else 0.0
        edge_f1 = (
            2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)
            if edge_precision + edge_recall
            else 0.0
        )
        aggregate: dict[str, object] = {
            "Method": method,
            "Evaluation Schema": "execution_provenance_span_v2",
            "Path Recall@10": (
                sum(path_recall_values) / len(path_recall_values)
                if path_recall_values
                else "N/A"
            ),
            "Edge Recall@10": edge_recall if gold_edges else "N/A",
            "Edge Precision@10": edge_precision if gold_edges else "N/A",
            "Edge F1@10": edge_f1 if gold_edges else "N/A",
            "Index Build Time": 0.0,
            "Graph Construction Time": 0.0,
            "Query-Evidence Connectivity@10": "N/A",
        }
        rows = [row.model_dump(mode="python", by_alias=True) for row in task_rows]
        for field_name, field in TaskMetricRow.model_fields.items():
            alias = field.serialization_alias or field_name
            if alias in aggregate:
                continue
            aggregate[alias] = sum(float(row[alias]) for row in rows) / len(rows)
        return [MetricRow.model_validate(aggregate)], per_task

    def build_failure_cases(
        self,
        request: SpanEvidenceEvaluationRequest,
        *,
        top_k: int,
        limit: int,
    ) -> list[FailureCase]:
        if limit <= 0:
            return []
        labels = {label.task_id: label for label in request.labels}
        cases: list[FailureCase] = []
        for prediction in request.predictions:
            label = labels[prediction.task_id]
            missing = missing_gold_spans(
                prediction.ranked_nodes,
                label.gold_evidence_spans,
                top_k,
            )
            if not missing:
                continue
            cases.append(
                FailureCase(
                    debug_type="span_failure_case",
                    task_id=prediction.task_id,
                    method=prediction.method,
                    failure_type=f"missing_full_span_support_at_{top_k}",
                    gold_evidence_item_ids=tuple(
                        _span_id(span) for span in label.gold_evidence_spans
                    ),
                    retrieved_top_k=prediction.ranked_node_ids[:top_k],
                    missing_gold_nodes=tuple(_span_id(span) for span in missing),
                    connected_gold_in_top_k=False,
                )
            )
            if len(cases) >= limit:
                break
        return cases


def _span_id(span) -> str:
    return (
        f"{span.event_id}{span.json_pointer}:"
        f"{span.char_start}-{span.char_end}"
    )


__all__ = ["CONTEXT_TOKEN_BUDGET", "SpanEvidenceMetricSuite"]
