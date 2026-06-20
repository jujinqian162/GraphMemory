from __future__ import annotations

from typing import Protocol

from abstraction.domain.retrieval.predictions import (
    AnswerPrediction,
    ContextPrediction,
    RankingPrediction,
)
from abstraction.domain.retrieval.requests import AnswerRequest, ContextGatheringRequest
from abstraction.domain.retrieval.scoring import RankedScores


class RankingPredictionBuilder(Protocol):
    def build_ranking_prediction(self, ranked_scores: RankedScores) -> RankingPrediction:
        ...


class ContextPredictionBuilder(Protocol):
    def build_context_prediction(
        self,
        request: ContextGatheringRequest,
        ranked_scores: RankedScores,
    ) -> ContextPrediction:
        ...


class AnswerPredictionBuilder(Protocol):
    def build_answer_prediction(self, request: AnswerRequest, answer_text: str) -> AnswerPrediction:
        ...


class RankedScoresRankingPredictionBuilder:  # implement RankingPredictionBuilder
    def build_ranking_prediction(self, ranked_scores: RankedScores) -> RankingPrediction:
        raise NotImplementedError
class RankedScoresContextPredictionBuilder:  # implement ContextPredictionBuilder
    def build_context_prediction(
        self,
        request: ContextGatheringRequest,
        ranked_scores: RankedScores,
    ) -> ContextPrediction:
        raise NotImplementedError
class ReaderAnswerPredictionBuilder:  # implement AnswerPredictionBuilder
    def build_answer_prediction(self, request: AnswerRequest, answer_text: str) -> AnswerPrediction:
        raise NotImplementedError