from __future__ import annotations

from graph_memory.evaluation.contracts import MetricRow
from graph_memory.evaluation.requests import EvidenceEvaluationRequest
from graph_memory.evaluation.suites import evaluate_evidence


def evaluate_results(request: EvidenceEvaluationRequest) -> list[MetricRow]:
    return evaluate_evidence(request)


__all__ = ["evaluate_results"]