from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from graph_memory.contracts.common import EdgeType, JsonValue
from graph_memory.contracts.graphs import GraphItemNode, GraphNode, MemoryGraph
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import prepare_graph_input
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.graphs.construction.rules.bridge import BridgeEdgeRule
from graph_memory.graphs.construction.rules.contracts import GraphEdgeRule
from graph_memory.graphs.construction.rules.entity_overlap import EntityOverlapEdgeRule
from graph_memory.graphs.construction.rules.query_overlap import QueryOverlapEdgeRule
from graph_memory.graphs.construction.rules.sequential import SequentialEdgeRule
from graph_memory.graphs.requests import GraphBuildNode, GraphBuildRequest


@dataclass(frozen=True)
class GraphBuilder:
    config: GraphBuildConfig
    rules: tuple[GraphEdgeRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.rules:
            object.__setattr__(self, "rules", default_graph_edge_rules(self.config))

    def build(self, request: GraphBuildRequest) -> MemoryGraph:
        prepared_input = prepare_graph_input(request, self.config)
        nodes: list[GraphNode] = [
            {"id": "q", "node_type": "question", "text": request.query_text},
            *[_graph_item_node(node) for node in request.nodes],
        ]
        accumulator = EdgeAccumulator()
        for edge in request.input_visible_edges:
            accumulator.add(edge.source, edge.target, cast(EdgeType, edge.edge_type), edge.weight, directed=edge.directed)
        for rule in self.rules:
            rule.add_edges(prepared_input, accumulator)
        return {
            "task_id": request.task_id,
            "nodes": nodes,
            "edges": accumulator.edges,
        }

    def build_many(self, requests: list[GraphBuildRequest]) -> list[MemoryGraph]:
        return [self.build(request) for request in requests]


def default_graph_edge_rules(config: GraphBuildConfig) -> tuple[GraphEdgeRule, ...]:
    return (
        SequentialEdgeRule(),
        QueryOverlapEdgeRule(config),
        EntityOverlapEdgeRule(config),
        BridgeEdgeRule(config),
    )


def build_graphs(requests: list[GraphBuildRequest], config: GraphBuildConfig) -> list[MemoryGraph]:
    return GraphBuilder(config).build_many(requests)


def _graph_item_node(node: GraphBuildNode) -> GraphItemNode:
    graph_node: GraphItemNode = {
        "id": node.node_id,
        "node_type": "graph_item",
        "node_kind": node.node_kind,
        "text": node.text,
    }
    if node.source_ref is not None:
        graph_node["source_ref"] = node.source_ref
    if node.group_key is not None:
        graph_node["group_key"] = node.group_key
    if node.sequence_index is not None:
        graph_node["sequence_index"] = node.sequence_index
    if node.metadata:
        graph_node["metadata"] = cast(dict[str, JsonValue], dict(node.metadata))
    return graph_node
