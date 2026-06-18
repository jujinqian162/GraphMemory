from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.evaluation.metrics import MetricRow, MetricResultTable
from abstraction.domain.projections.prediction_to_eval import (
    AnswerEvalUnit,
    EvidenceEvalUnit,
    MultiHopEvalUnit,
    SupportCoverageEvalUnit,
)


class EvidenceRecallMetric(Protocol):
    def compute_evidence_recall(self, eval_units: Sequence[EvidenceEvalUnit]) -> Sequence[MetricRow]:
        ...


class SupportCoverageMetric(Protocol):
    def compute_support_coverage(self, eval_units: Sequence[SupportCoverageEvalUnit]) -> Sequence[MetricRow]:
        ...


class MultiHopSupportMetric(Protocol):
    def compute_multihop_support(self, eval_units: Sequence[MultiHopEvalUnit]) -> Sequence[MetricRow]:
        ...


class AnswerQualityMetric(Protocol):
    def compute_answer_quality(self, eval_units: Sequence[AnswerEvalUnit]) -> Sequence[MetricRow]:
        ...


class MetricResultAggregator(Protocol):
    def aggregate_metric_rows(self, rows: Sequence[MetricRow]) -> MetricResultTable:
        ...


class EvidenceRecallAtKMetric:  # implement EvidenceRecallMetric
    def compute_evidence_recall(self, eval_units: Sequence[EvidenceEvalUnit]) -> Sequence[MetricRow]:
        pass


class TurnSessionSupportMetric:  # implement SupportCoverageMetric
    def compute_support_coverage(self, eval_units: Sequence[SupportCoverageEvalUnit]) -> Sequence[MetricRow]:
        pass


class ParagraphEntityPathMetric:  # implement MultiHopSupportMetric
    def compute_multihop_support(self, eval_units: Sequence[MultiHopEvalUnit]) -> Sequence[MetricRow]:
        pass


class ExactMatchF1AnswerMetric:  # implement AnswerQualityMetric
    def compute_answer_quality(self, eval_units: Sequence[AnswerEvalUnit]) -> Sequence[MetricRow]:
        pass


class PerTaskMetricResultAggregator:  # implement MetricResultAggregator
    def aggregate_metric_rows(self, rows: Sequence[MetricRow]) -> MetricResultTable:
        pass

