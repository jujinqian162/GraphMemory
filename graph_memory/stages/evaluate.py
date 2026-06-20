from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import FailureCase, MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToEvidenceEvaluationRequest
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord
from graph_memory.evaluation.suites import evidence_metric_suite
from graph_memory.registry.stage_configs import EvaluateStageConfig


@dataclass(frozen=True)
class EvaluateStageResult:
    metric_rows: list[MetricRow]
    failure_cases: list[FailureCase]


def run_evaluate_stage(
    config: EvaluateStageConfig,
    *,
    predictions: list[RankedResult],
    labels: list[HotpotQALabelRecord],
    graphs: list[MemoryGraph],
) -> EvaluateStageResult:
    request = HotpotQAToEvidenceEvaluationRequest().project(
        predictions=predictions,
        labels=labels,
        graphs=graphs,
    )
    suite = evidence_metric_suite()
    metric_rows = suite.evaluate(request)
    failure_cases = suite.build_failure_cases(
        request,
        top_k=10,
        limit=config.failure_case_limit,
    )
    return EvaluateStageResult(metric_rows=metric_rows, failure_cases=failure_cases)


__all__ = ["EvaluateStageResult", "run_evaluate_stage"]
