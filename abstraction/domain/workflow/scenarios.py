from __future__ import annotations

from typing import Protocol

from abstraction.domain.common.capability_names import RequestKind, ViewKind
from abstraction.domain.projections.ports import ProjectionRegistry
from abstraction.domain.retrieval.capabilities import MethodCapability


class ScenarioFlow(Protocol):
    def describe_required_view(self) -> ViewKind:
        ...

    def describe_required_request(self) -> RequestKind:
        ...

    def connect_projection(self, registry: ProjectionRegistry) -> None:
        ...


class DenseLongMemEvalScenarioFlow:  # implement ScenarioFlow
    def describe_required_view(self) -> ViewKind:
        raise NotImplementedError
    def describe_required_request(self) -> RequestKind:
        raise NotImplementedError
    def connect_projection(self, registry: ProjectionRegistry) -> None:
        raise NotImplementedError
class GraphRerankLongMemEvalScenarioFlow:  # implement ScenarioFlow
    def describe_required_view(self) -> ViewKind:
        raise NotImplementedError
    def describe_required_request(self) -> RequestKind:
        raise NotImplementedError
    def connect_projection(self, registry: ProjectionRegistry) -> None:
        raise NotImplementedError
class GraphRAGTwoWikiScenarioFlow:  # implement ScenarioFlow
    def describe_required_view(self) -> ViewKind:
        raise NotImplementedError
    def describe_required_request(self) -> RequestKind:
        raise NotImplementedError
    def connect_projection(self, registry: ProjectionRegistry) -> None:
        raise NotImplementedError
class HotpotEvidenceRankingScenarioFlow:  # implement ScenarioFlow
    def describe_required_view(self) -> ViewKind:
        raise NotImplementedError
    def describe_required_request(self) -> RequestKind:
        raise NotImplementedError
    def connect_projection(self, registry: ProjectionRegistry) -> None:
        raise NotImplementedError
class ScenarioCompatibilityReviewer:
    def assert_method_matches_scenario(
        self,
        scenario: ScenarioFlow,
        method_capability: MethodCapability,
    ) -> None:
        raise NotImplementedError