from __future__ import annotations

from typing import Protocol

from abstraction.domain.common.identifiers import ArtifactId
from abstraction.domain.graphs.artifacts import GraphArtifact, GraphIndexView, GraphNode
from abstraction.domain.graphs.rules import GraphRuleSet
from abstraction.domain.task_views.views import GraphBuildView


class GraphBuilder(Protocol):
    def build_graph(self, view: GraphBuildView, rule_set: GraphRuleSet) -> GraphArtifact:
        ...


class GraphIndexBuilder(Protocol):
    def build_graph_index(self, artifact: GraphArtifact) -> GraphIndexView:
        ...


class RuleSetGraphBuilder:  # implement GraphBuilder
    def build_graph(self, view: GraphBuildView, rule_set: GraphRuleSet) -> GraphArtifact:
        rule_name = rule_set.describe_rule_set()
        candidate_nodes = [
            GraphNode(
                item_id=node_view.item_id,
                node_kind="candidate",
                visible_metadata=node_view.grouping_metadata,
            )
            for node_view in view.candidate_nodes
        ]
        input_visible_edges = rule_set.derive_visible_edges(view)
        return GraphArtifact(
            artifact_id=ArtifactId(value=f"{view.task_id.value}:{rule_name}"),
            task_id=view.task_id,
            nodes=candidate_nodes,
            edges=input_visible_edges,
            graph_metadata={"rule_set": rule_name},
        )


class ArtifactGraphIndexBuilder:  # implement GraphIndexBuilder
    def build_graph_index(self, artifact: GraphArtifact) -> GraphIndexView:
        return GraphIndexView(
            graph_ref=artifact.artifact_id.value,
            task_id=artifact.task_id,
            candidate_item_ids=[node.item_id for node in artifact.nodes],
            edge_kinds=sorted({edge.edge_kind for edge in artifact.edges}),
        )
