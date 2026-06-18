from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.common.identifiers import MetricSuiteId
from abstraction.domain.evaluation.metrics import MetricResultTable, MetricSelectionKey
from abstraction.domain.evaluation.primitives import (
    AnswerQualityMetric,
    EvidenceRecallMetric,
    MetricResultAggregator,
    MultiHopSupportMetric,
    SupportCoverageMetric,
)
from abstraction.domain.projections.eval_units import (
    AnswerEvalUnit,
    EvidenceEvalUnit,
    EvaluationUnitBatch,
    MultiHopEvalUnit,
    SupportCoverageEvalUnit,
)


class MetricSuite(Protocol):
    def describe_metric_suite(self) -> MetricSelectionKey:
        ...

    def evaluate_units(self, eval_units: EvaluationUnitBatch) -> MetricResultTable:
        ...


class MetricSuiteRegistry(Protocol):
    def register_metric_suite(self, suite: MetricSuite) -> None:
        ...

    def get_metric_suite(self, metric_suite_id: MetricSuiteId) -> MetricSuite:
        ...


class EvidenceMetricSuite:  # implement MetricSuite
    def __init__(
        self,
        evidence_recall_metric: EvidenceRecallMetric,
        result_aggregator: MetricResultAggregator,
    ) -> None:
        self.evidence_recall_metric = evidence_recall_metric
        self.result_aggregator = result_aggregator

    def describe_metric_suite(self) -> MetricSelectionKey:
        pass

    def evaluate_units(self, eval_units: EvaluationUnitBatch) -> MetricResultTable:
        evidence_units = [unit for unit in eval_units if isinstance(unit, EvidenceEvalUnit)]
        metric_rows = self.evidence_recall_metric.compute_evidence_recall(evidence_units)
        return self.result_aggregator.aggregate_metric_rows(metric_rows)


class LongMemEvalMetricSuite:  # implement MetricSuite
    def __init__(
        self,
        support_coverage_metric: SupportCoverageMetric,
        answer_quality_metric: AnswerQualityMetric,
        result_aggregator: MetricResultAggregator,
    ) -> None:
        self.support_coverage_metric = support_coverage_metric
        self.answer_quality_metric = answer_quality_metric
        self.result_aggregator = result_aggregator

    def describe_metric_suite(self) -> MetricSelectionKey:
        pass

    def evaluate_units(self, eval_units: EvaluationUnitBatch) -> MetricResultTable:
        support_units = [unit for unit in eval_units if isinstance(unit, SupportCoverageEvalUnit)]
        answer_units = [unit for unit in eval_units if isinstance(unit, AnswerEvalUnit)]
        support_rows = self.support_coverage_metric.compute_support_coverage(support_units)
        answer_rows = self.answer_quality_metric.compute_answer_quality(answer_units)
        return self.result_aggregator.aggregate_metric_rows([*support_rows, *answer_rows])


class MultiHopMetricSuite:  # implement MetricSuite
    def __init__(
        self,
        evidence_recall_metric: EvidenceRecallMetric,
        multihop_support_metric: MultiHopSupportMetric,
        answer_quality_metric: AnswerQualityMetric,
        result_aggregator: MetricResultAggregator,
    ) -> None:
        self.evidence_recall_metric = evidence_recall_metric
        self.multihop_support_metric = multihop_support_metric
        self.answer_quality_metric = answer_quality_metric
        self.result_aggregator = result_aggregator

    def describe_metric_suite(self) -> MetricSelectionKey:
        pass

    def evaluate_units(self, eval_units: EvaluationUnitBatch) -> MetricResultTable:
        evidence_units = [unit for unit in eval_units if isinstance(unit, EvidenceEvalUnit)]
        multihop_units = [unit for unit in eval_units if isinstance(unit, MultiHopEvalUnit)]
        answer_units = [unit for unit in eval_units if isinstance(unit, AnswerEvalUnit)]
        evidence_rows = self.evidence_recall_metric.compute_evidence_recall(evidence_units)
        multihop_rows = self.multihop_support_metric.compute_multihop_support(multihop_units)
        answer_rows = self.answer_quality_metric.compute_answer_quality(answer_units)
        return self.result_aggregator.aggregate_metric_rows(
            [
                *evidence_rows,
                *multihop_rows,
                *answer_rows,
            ]
        )


class CapabilityMetricSuiteRegistry:  # implement MetricSuiteRegistry
    def __init__(self) -> None:
        self.suite_by_metric_suite_id: dict[MetricSuiteId, MetricSuite] = {}

    def register_metric_suite(self, suite: MetricSuite) -> None:
        metric_key = suite.describe_metric_suite()
        self.suite_by_metric_suite_id[metric_key.metric_suite_id] = suite

    def get_metric_suite(self, metric_suite_id: MetricSuiteId) -> MetricSuite:
        return self.suite_by_metric_suite_id[metric_suite_id]
