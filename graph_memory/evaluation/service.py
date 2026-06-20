from __future__ import annotations

from graph_memory.contracts.metrics import MetricRow
from graph_memory.evaluation.requests import EvidenceEvaluationRequest
from graph_memory.evaluation.suites import evidence_metric_suite


def evaluate_results(request: EvidenceEvaluationRequest) -> list[MetricRow]:
    return evidence_metric_suite().evaluate(request)


__all__ = ["evaluate_results"]