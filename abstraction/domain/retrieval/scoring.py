from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from abstraction.domain.common.identifiers import ItemId
from abstraction.domain.retrieval.requests import (
    ContextGatheringRequest,
    GraphRankingRequest,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class ScoreVector:
    score_name: str
    score_by_item: Mapping[ItemId, float]


@dataclass(frozen=True)
class ScoreWeight:
    score_name: str
    weight: float


@dataclass(frozen=True)
class RankedScores:
    ranked_item_ids: Sequence[ItemId]
    final_score_by_item: Mapping[ItemId, float]
    component_scores: Sequence[ScoreVector]


class TextRelevanceScorer(Protocol):
    def score_text_relevance(self, request: TextRankingRequest) -> ScoreVector:
        ...


class GraphSupportScorer(Protocol):
    def score_graph_support(self, request: GraphRankingRequest) -> ScoreVector:
        ...


class QueryBridgeScorer(Protocol):
    def score_query_bridge(self, request: GraphRankingRequest) -> ScoreVector:
        ...


class TemporalRecencyScorer(Protocol):
    def score_recency(self, request: TemporalMemoryRankingRequest) -> ScoreVector:
        ...


class MemoryImportanceScorer(Protocol):
    def score_importance(self, request: TemporalMemoryRankingRequest) -> ScoreVector:
        ...


class ContextExpansionScorer(Protocol):
    def score_context_expansion(self, request: ContextGatheringRequest) -> ScoreVector:
        ...


class ScoreNormalizer(Protocol):
    def normalize_scores(self, scores: Sequence[ScoreVector]) -> Sequence[ScoreVector]:
        ...


class WeightedScoreCombiner(Protocol):
    def combine_scores(
        self,
        scores: Sequence[ScoreVector],
        weights: Sequence[ScoreWeight],
        candidate_item_ids: Sequence[ItemId],
    ) -> RankedScores:
        ...


class BM25TextRelevanceScorer:  # implement TextRelevanceScorer
    def score_text_relevance(self, request: TextRankingRequest) -> ScoreVector:
        raise NotImplementedError
class DenseTextRelevanceScorer:  # implement TextRelevanceScorer
    def score_text_relevance(self, request: TextRankingRequest) -> ScoreVector:
        raise NotImplementedError
class GraphTopologySupportScorer:  # implement GraphSupportScorer
    def score_graph_support(self, request: GraphRankingRequest) -> ScoreVector:
        raise NotImplementedError
class GraphQueryBridgeScorer:  # implement QueryBridgeScorer
    def score_query_bridge(self, request: GraphRankingRequest) -> ScoreVector:
        raise NotImplementedError
class PositionRecencyScorer:  # implement TemporalRecencyScorer
    def score_recency(self, request: TemporalMemoryRankingRequest) -> ScoreVector:
        raise NotImplementedError
class SidecarImportanceScorer:  # implement MemoryImportanceScorer
    def score_importance(self, request: TemporalMemoryRankingRequest) -> ScoreVector:
        raise NotImplementedError
class GraphContextExpansionScorer:  # implement ContextExpansionScorer
    def score_context_expansion(self, request: ContextGatheringRequest) -> ScoreVector:
        raise NotImplementedError
class PerTaskScoreNormalizer:  # implement ScoreNormalizer
    def normalize_scores(self, scores: Sequence[ScoreVector]) -> Sequence[ScoreVector]:
        raise NotImplementedError
class LinearWeightedScoreCombiner:  # implement WeightedScoreCombiner
    def combine_scores(
        self,
        scores: Sequence[ScoreVector],
        weights: Sequence[ScoreWeight],
        candidate_item_ids: Sequence[ItemId],
    ) -> RankedScores:
        raise NotImplementedError