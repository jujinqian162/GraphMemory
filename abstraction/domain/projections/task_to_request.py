from __future__ import annotations

from abstraction.domain.projections.ports import ProjectionDefinition
from abstraction.domain.retrieval.requests import (
    ContextGatheringRequest,
    GraphRankingRequest,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)
from abstraction.domain.task_views.views import (
    ContextGatheringView,
    EvidenceRankingView,
    GraphBuildView,
)


class EvidenceToTextRankingProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: EvidenceRankingView) -> TextRankingRequest:
        pass


class ContextToTextRankingProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: ContextGatheringView) -> TextRankingRequest:
        pass


class GraphBuildToGraphRankingProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: GraphBuildView) -> GraphRankingRequest:
        pass


class ContextToTemporalMemoryProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: ContextGatheringView) -> TemporalMemoryRankingRequest:
        pass


class ContextToContextGatheringProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: ContextGatheringView) -> ContextGatheringRequest:
        pass


class EvidenceGraphToContextGatheringProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        pass

    def project(self, source: GraphBuildView) -> ContextGatheringRequest:
        pass

