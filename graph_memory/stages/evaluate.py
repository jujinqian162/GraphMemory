from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import FailureCase, MetricTableSchema, SuiteMetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.selection import evidence_evaluation_request_for_dataset
from graph_memory.evaluation.suites import MetricSuite, metric_suite_for_dataset
from graph_memory.registry.stage_configs import EvaluateStageConfig


@dataclass(frozen=True)
class EvaluateStageResult:
    metric_rows: list[SuiteMetricRow]
    failure_cases: list[FailureCase]
    metric_suite: MetricSuite
    metric_suite_name: str
    metric_table_schema: MetricTableSchema


def run_evaluate_stage(
    config: EvaluateStageConfig,
    *,
    predictions: list[RankedResult],
    labels: list[object],
    graphs: list[MemoryGraph],
) -> EvaluateStageResult:
    request = evidence_evaluation_request_for_dataset(
        config.dataset,
        predictions=predictions,
        labels=labels,
        graphs=graphs,
    )
    suite = metric_suite_for_dataset(config.dataset)
    metric_rows = list(suite.evaluate(request))
    failure_cases = suite.build_failure_cases(
        request,
        top_k=10,
        limit=config.failure_case_limit,
    )
    return EvaluateStageResult(
        metric_rows=metric_rows,
        failure_cases=failure_cases,
        metric_suite=suite,
        metric_suite_name=suite.name,
        metric_table_schema=suite.table_schema,
    )


__all__ = ["EvaluateStageResult", "run_evaluate_stage"]
