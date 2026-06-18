from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.graphs.artifacts import GraphEdge
from abstraction.domain.task_views.views import GraphBuildView


class GraphRuleSet(Protocol):
    def describe_rule_set(self) -> str:
        ...

    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        ...


class HotpotQAGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        pass

    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        pass


class LongMemEvalGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        pass

    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        pass


class TwoWikiGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        pass

    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        pass


class GraphRAGInputGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        pass

    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        pass
