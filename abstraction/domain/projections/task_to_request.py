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
        raise NotImplementedError
    def project(self, source: EvidenceRankingView) -> TextRankingRequest:
        raise NotImplementedError
class ContextToTextRankingProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: ContextGatheringView) -> TextRankingRequest:
        raise NotImplementedError
class GraphBuildToGraphRankingProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: GraphBuildView) -> GraphRankingRequest:
        raise NotImplementedError
class ContextToTemporalMemoryProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: ContextGatheringView) -> TemporalMemoryRankingRequest:
        raise NotImplementedError
class ContextToContextGatheringProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: ContextGatheringView) -> ContextGatheringRequest:
        raise NotImplementedError
class EvidenceGraphToContextGatheringProjection:  # implement ProjectionAdapter
    def describe_projection(self) -> ProjectionDefinition:
        raise NotImplementedError
    def project(self, source: GraphBuildView) -> ContextGatheringRequest:
        raise NotImplementedError