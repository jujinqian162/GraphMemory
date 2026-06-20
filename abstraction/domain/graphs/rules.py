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
        raise NotImplementedError
    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        raise NotImplementedError
class LongMemEvalGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        raise NotImplementedError
    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        raise NotImplementedError
class TwoWikiGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        raise NotImplementedError
    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        raise NotImplementedError
class GraphRAGInputGraphRuleSet:  # implement GraphRuleSet
    def describe_rule_set(self) -> str:
        raise NotImplementedError
    def derive_visible_edges(self, view: GraphBuildView) -> Sequence[GraphEdge]:
        raise NotImplementedError