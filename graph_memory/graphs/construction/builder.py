from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tqdm.auto import tqdm

from graph_memory.contracts.common import EdgeType
from graph_memory.graphs.contracts import GraphItemNode, GraphNode, EvidenceGraph, QuestionNode
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import prepare_graph_input
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.graphs.construction.rules.bridge import BridgeEdgeRule
from graph_memory.graphs.construction.rules.contracts import GraphEdgeRule
from graph_memory.graphs.construction.rules.entity_overlap import EntityOverlapEdgeRule
from graph_memory.graphs.construction.rules.query_overlap import QueryOverlapEdgeRule
from graph_memory.graphs.construction.rules.sequential import SequentialEdgeRule
from graph_memory.graphs.requests import (
    EvidenceGraphBuildNode,
    EvidenceGraphBuildRequest,
)


@dataclass(frozen=True)
class GraphBuilder:
    config: GraphBuildConfig
    rules: tuple[GraphEdgeRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.rules:
            object.__setattr__(self, "rules", default_graph_edge_rules(self.config))

    def build(self, request: EvidenceGraphBuildRequest) -> EvidenceGraph:
        prepared_input = prepare_graph_input(request, self.config)
        nodes: list[GraphNode] = [
            QuestionNode(text=request.query_text),
            *[_graph_item_node(node) for node in request.nodes],
        ]
        accumulator = EdgeAccumulator()
        for edge in request.input_visible_edges:
            accumulator.add(
                edge.source,
                edge.target,
                cast(EdgeType, edge.edge_type),
                edge.weight,
                directed=edge.directed,
            )
        for rule in self.rules:
            rule.add_edges(prepared_input, accumulator)
        return EvidenceGraph(
            task_id=request.task_id,
            nodes=tuple(nodes),
            edges=tuple(accumulator.edges),
        )

    def build_many(
        self,
        requests: list[EvidenceGraphBuildRequest],
        *,
        progress_desc: str | None = None,
    ) -> list[EvidenceGraph]:
        iterator = requests
        if progress_desc is not None:
            iterator = tqdm(requests, desc=progress_desc, unit="graph")
        return [self.build(request) for request in iterator]


def default_graph_edge_rules(config: GraphBuildConfig) -> tuple[GraphEdgeRule, ...]:
    return (
        SequentialEdgeRule(),
        QueryOverlapEdgeRule(config),
        EntityOverlapEdgeRule(config),
        BridgeEdgeRule(config),
    )


def build_graphs(
    requests: list[EvidenceGraphBuildRequest],
    config: GraphBuildConfig,
    *,
    progress_desc: str | None = None,
) -> list[EvidenceGraph]:
    return GraphBuilder(config).build_many(requests, progress_desc=progress_desc)


def _graph_item_node(node: EvidenceGraphBuildNode) -> GraphItemNode:
    return GraphItemNode(
        id=node.node_id,
        node_kind=node.node_kind,
        text=node.text,
        source_ref=node.source_ref,
        group_key=node.group_key,
        sequence_index=node.sequence_index,
        metadata=dict(node.metadata) or None,
    )
