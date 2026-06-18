from __future__ import annotations

from typing import Protocol

from abstraction.domain.retrieval.builders import (
    AnswerPredictionBuilder,
    ContextPredictionBuilder,
    RankingPredictionBuilder,
)
from abstraction.domain.retrieval.predictions import (
    AnswerPrediction,
    ContextPrediction,
    RankingPrediction,
)
from abstraction.domain.retrieval.requests import (
    AnswerRequest,
    ContextGatheringRequest,
    GraphRankingRequest,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)
from abstraction.domain.retrieval.scoring import (
    ContextExpansionScorer,
    GraphSupportScorer,
    MemoryImportanceScorer,
    QueryBridgeScorer,
    ScoreNormalizer,
    ScoreVector,
    ScoreWeight,
    TemporalRecencyScorer,
    TextRelevanceScorer,
    WeightedScoreCombiner,
)


class TextRankingMethod(Protocol):
    def rank_task(self, request: TextRankingRequest) -> RankingPrediction:
        ...


class GraphRankingMethod(Protocol):
    def rank_task(self, request: GraphRankingRequest) -> RankingPrediction:
        ...


class TemporalMemoryRankingMethod(Protocol):
    def rank_task(self, request: TemporalMemoryRankingRequest) -> RankingPrediction:
        ...


class ContextGatheringMethod(Protocol):
    def gather_task_context(self, request: ContextGatheringRequest) -> ContextPrediction:
        ...


class AnsweringMethod(Protocol):
    def answer_task(self, request: AnswerRequest) -> AnswerPrediction:
        ...


class BM25RankingMethod:  # implement TextRankingMethod
    def __init__(
        self,
        relevance_scorer: TextRelevanceScorer,
        score_normalizer: ScoreNormalizer,
        score_combiner: WeightedScoreCombiner,
        prediction_builder: RankingPredictionBuilder,
    ) -> None:
        self.relevance_scorer = relevance_scorer
        self.score_normalizer = score_normalizer
        self.score_combiner = score_combiner
        self.prediction_builder = prediction_builder

    def rank_task(self, request: TextRankingRequest) -> RankingPrediction:
        relevance_scores = self.relevance_scorer.score_text_relevance(request)
        normalized_scores = self.score_normalizer.normalize_scores([relevance_scores])
        ranked_scores = self.score_combiner.combine_scores(
            scores=normalized_scores,
            weights=[ScoreWeight(score_name="text_relevance", weight=1.0)],
            candidate_item_ids=list(request.candidate_text_by_item),
        )
        return self.prediction_builder.build_ranking_prediction(ranked_scores)


class DenseRankingMethod:  # implement TextRankingMethod
    def __init__(
        self,
        relevance_scorer: TextRelevanceScorer,
        score_normalizer: ScoreNormalizer,
        score_combiner: WeightedScoreCombiner,
        prediction_builder: RankingPredictionBuilder,
    ) -> None:
        self.relevance_scorer = relevance_scorer
        self.score_normalizer = score_normalizer
        self.score_combiner = score_combiner
        self.prediction_builder = prediction_builder

    def rank_task(self, request: TextRankingRequest) -> RankingPrediction:
        relevance_scores = self.relevance_scorer.score_text_relevance(request)
        normalized_scores = self.score_normalizer.normalize_scores([relevance_scores])
        ranked_scores = self.score_combiner.combine_scores(
            scores=normalized_scores,
            weights=[ScoreWeight(score_name="text_relevance", weight=1.0)],
            candidate_item_ids=list(request.candidate_text_by_item),
        )
        return self.prediction_builder.build_ranking_prediction(ranked_scores)


class DenseFineTunedRankingMethod(DenseRankingMethod):  # implement TextRankingMethod
    pass


class GraphRerankMethod:  # implement GraphRankingMethod
    def __init__(
        self,
        graph_support_scorer: GraphSupportScorer,
        query_bridge_scorer: QueryBridgeScorer,
        score_normalizer: ScoreNormalizer,
        score_combiner: WeightedScoreCombiner,
        prediction_builder: RankingPredictionBuilder,
        graph_weights: tuple[ScoreWeight, ...],
    ) -> None:
        self.graph_support_scorer = graph_support_scorer
        self.query_bridge_scorer = query_bridge_scorer
        self.score_normalizer = score_normalizer
        self.score_combiner = score_combiner
        self.prediction_builder = prediction_builder
        self.graph_weights = graph_weights

    def rank_task(self, request: GraphRankingRequest) -> RankingPrediction:
        seed_score_vector = ScoreVector(score_name="seed_score", score_by_item=request.seed_scores_by_item)
        graph_support_scores = self.graph_support_scorer.score_graph_support(request)
        query_bridge_scores = self.query_bridge_scorer.score_query_bridge(request)
        normalized_scores = self.score_normalizer.normalize_scores(
            [
                seed_score_vector,
                graph_support_scores,
                query_bridge_scores,
            ]
        )
        ranked_scores = self.score_combiner.combine_scores(
            scores=normalized_scores,
            weights=list(self.graph_weights),
            candidate_item_ids=request.candidate_item_ids,
        )
        return self.prediction_builder.build_ranking_prediction(ranked_scores)


class TrainableGraphRankingMethod(GraphRerankMethod):  # implement GraphRankingMethod
    pass


class MemoryStreamRankingMethod:  # implement TemporalMemoryRankingMethod
    def __init__(
        self,
        relevance_scorer: TextRelevanceScorer,
        recency_scorer: TemporalRecencyScorer,
        importance_scorer: MemoryImportanceScorer,
        score_normalizer: ScoreNormalizer,
        score_combiner: WeightedScoreCombiner,
        prediction_builder: RankingPredictionBuilder,
        memory_weights: tuple[ScoreWeight, ...],
    ) -> None:
        self.relevance_scorer = relevance_scorer
        self.recency_scorer = recency_scorer
        self.importance_scorer = importance_scorer
        self.score_normalizer = score_normalizer
        self.score_combiner = score_combiner
        self.prediction_builder = prediction_builder
        self.memory_weights = memory_weights

    def rank_task(self, request: TemporalMemoryRankingRequest) -> RankingPrediction:
        text_request = TextRankingRequest(
            request_id=request.request_id,
            request_kind=request.request_kind,
            task_id=request.task_id,
            query_text=request.query_text,
            candidate_text_by_item=request.memory_text_by_item,
        )
        relevance_scores = self.relevance_scorer.score_text_relevance(text_request)
        recency_scores = self.recency_scorer.score_recency(request)
        importance_scores = self.importance_scorer.score_importance(request)
        normalized_scores = self.score_normalizer.normalize_scores(
            [
                relevance_scores,
                recency_scores,
                importance_scores,
            ]
        )
        ranked_scores = self.score_combiner.combine_scores(
            scores=normalized_scores,
            weights=list(self.memory_weights),
            candidate_item_ids=request.memory_item_ids,
        )
        return self.prediction_builder.build_ranking_prediction(ranked_scores)


class GraphRAGContextMethod:  # implement ContextGatheringMethod
    def __init__(
        self,
        seed_relevance_scorer: TextRelevanceScorer,
        context_expansion_scorer: ContextExpansionScorer,
        score_normalizer: ScoreNormalizer,
        score_combiner: WeightedScoreCombiner,
        prediction_builder: ContextPredictionBuilder,
        context_weights: tuple[ScoreWeight, ...],
    ) -> None:
        self.seed_relevance_scorer = seed_relevance_scorer
        self.context_expansion_scorer = context_expansion_scorer
        self.score_normalizer = score_normalizer
        self.score_combiner = score_combiner
        self.prediction_builder = prediction_builder
        self.context_weights = context_weights

    def gather_task_context(self, request: ContextGatheringRequest) -> ContextPrediction:
        text_request = TextRankingRequest(
            request_id=request.request_id,
            request_kind=request.request_kind,
            task_id=request.task_id,
            query_text=request.question_text,
            candidate_text_by_item=request.candidate_text_by_item,
        )
        seed_relevance_scores = self.seed_relevance_scorer.score_text_relevance(text_request)
        context_expansion_scores = self.context_expansion_scorer.score_context_expansion(request)
        normalized_scores = self.score_normalizer.normalize_scores(
            [
                seed_relevance_scores,
                context_expansion_scores,
            ]
        )
        ranked_scores = self.score_combiner.combine_scores(
            scores=normalized_scores,
            weights=list(self.context_weights),
            candidate_item_ids=request.candidate_context_item_ids,
        )
        return self.prediction_builder.build_context_prediction(request, ranked_scores)


class ReaderAnsweringMethod:  # implement AnsweringMethod
    def __init__(self, prediction_builder: AnswerPredictionBuilder) -> None:
        self.prediction_builder = prediction_builder

    def answer_task(self, request: AnswerRequest) -> AnswerPrediction:
        answer_text = self._draft_answer(request)
        return self.prediction_builder.build_answer_prediction(request, answer_text)

    def _draft_answer(self, request: AnswerRequest) -> str:
        pass
