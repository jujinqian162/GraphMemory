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
                    "MRR": span_mrr(prediction.ranked_nodes, label.gold_evidence_spans),
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
                    **row.model_dump(mode="python"),
                )
            )
        aggregate: dict[str, object] = {
            "Method": method,
            "Evaluation Schema": "execution_provenance_span_v7",
            "Path Recall@10": "N/A",
            "Edge Recall@10": "N/A",
            "Edge Precision@10": "N/A",
            "Edge F1@10": "N/A",
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
    return f"{span.event_id}{span.json_pointer}:{span.char_start}-{span.char_end}"


__all__ = ["CONTEXT_TOKEN_BUDGET", "SpanEvidenceMetricSuite"]
