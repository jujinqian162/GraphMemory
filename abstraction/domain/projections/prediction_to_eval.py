from __future__ import annotations

from abstraction.domain.projections.eval_units import (
    AnswerEvalUnit,
    EvidenceEvalUnit,
    EvaluationUnitBatch,
    MultiHopEvalUnit,
    SupportCoverageEvalUnit,
)
from abstraction.domain.projections.ports import ProjectionDefinition
from abstraction.domain.retrieval.predictions import (
    AnswerPrediction,
    ContextPrediction,
    RankingPrediction,
)
from abstraction.domain.task_views.eval_views import (
    EvidenceEvalView,
    LongMemEvalEvalView,
    MultiHopEvalView,
)


class RankingToEvidenceEvalProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: tuple[RankingPrediction, EvidenceEvalView]) -> EvaluationUnitBatch:
        pass


class RankingToLongMemEvalSupportProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: tuple[RankingPrediction, LongMemEvalEvalView]) -> EvaluationUnitBatch:
        pass


class ContextToMultiHopEvalProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: tuple[ContextPrediction, MultiHopEvalView]) -> EvaluationUnitBatch:
        pass


class AnswerToAnswerEvalProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: tuple[AnswerPrediction, LongMemEvalEvalView | MultiHopEvalView]) -> EvaluationUnitBatch:
        pass
