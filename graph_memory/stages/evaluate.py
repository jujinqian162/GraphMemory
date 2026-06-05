from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import FailureCase, MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskLabels
from graph_memory.evaluation.failure_cases import build_failure_cases
from graph_memory.evaluation.service import evaluate_results
from graph_memory.registry.stage_configs import EvaluateStageConfig


@dataclass(frozen=True)
class EvaluateStageResult:
    metric_rows: list[MetricRow]
    failure_cases: list[FailureCase]


def run_evaluate_stage(
    config: EvaluateStageConfig,
    *,
    predictions: list[RankedResult],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
) -> EvaluateStageResult:
    metric_rows = evaluate_results(predictions, labels, graphs)
    failure_cases = build_failure_cases(
        predictions,
        labels,
        graphs,
        top_k=10,
        limit=config.failure_case_limit,
    )
    return EvaluateStageResult(metric_rows=metric_rows, failure_cases=failure_cases)


__all__ = ["EvaluateStageResult", "run_evaluate_stage"]
