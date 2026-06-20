from __future__ import annotations

from graph_memory.contracts.metrics import FailureCase
from graph_memory.evaluation.requests import EvidenceEvaluationRequest
from graph_memory.evaluation.suites import evidence_metric_suite


def build_failure_cases(
    request: EvidenceEvaluationRequest,
    *,
    top_k: int = 10,
    limit: int = 0,
) -> list[FailureCase]:
    return evidence_metric_suite().build_failure_cases(request, top_k=top_k, limit=limit)


__all__ = ["build_failure_cases"]