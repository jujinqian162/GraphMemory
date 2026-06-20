from __future__ import annotations

from abstraction.domain.projections.eval_units import EvaluationUnitBatch
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
        raise NotImplementedError
    def project(self, source: tuple[RankingPrediction, EvidenceEvalView]) -> EvaluationUnitBatch:
        raise NotImplementedError
class RankingToLongMemEvalSupportProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: tuple[RankingPrediction, LongMemEvalEvalView]) -> EvaluationUnitBatch:
        raise NotImplementedError
class ContextToMultiHopEvalProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: tuple[ContextPrediction, MultiHopEvalView]) -> EvaluationUnitBatch:
        raise NotImplementedError
class AnswerToAnswerEvalProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: tuple[AnswerPrediction, LongMemEvalEvalView | MultiHopEvalView]) -> EvaluationUnitBatch:
        raise NotImplementedError