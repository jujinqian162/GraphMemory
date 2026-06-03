from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import GraphNode, MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import prepare_graph_input
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.graphs.construction.rules.bridge import BridgeEdgeRule
from graph_memory.graphs.construction.rules.contracts import GraphEdgeRule
from graph_memory.graphs.construction.rules.entity_overlap import EntityOverlapEdgeRule
from graph_memory.graphs.construction.rules.query_overlap import QueryOverlapEdgeRule
from graph_memory.graphs.construction.rules.sequential import SequentialEdgeRule


@dataclass(frozen=True)
class GraphBuilder:
    config: GraphBuildConfig
    rules: tuple[GraphEdgeRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.rules:
            object.__setattr__(self, "rules", default_graph_edge_rules(self.config))

    def build(self, task_input: MemoryTaskInput) -> MemoryGraph:
        prepared_input = prepare_graph_input(task_input, self.config)
        nodes: list[GraphNode] = [
            {"id": "q", "node_type": "question", "text": task_input["query"]},
            *task_input["memory_items"],
        ]
        accumulator = EdgeAccumulator()
        for rule in self.rules:
            rule.add_edges(prepared_input, accumulator)
        return {
            "task_id": task_input["task_id"],
            "nodes": nodes,
            "edges": accumulator.edges,
        }

    def build_many(self, task_inputs: list[MemoryTaskInput]) -> list[MemoryGraph]:
        return [self.build(task_input) for task_input in task_inputs]


def default_graph_edge_rules(config: GraphBuildConfig) -> tuple[GraphEdgeRule, ...]:
    return (
        SequentialEdgeRule(),
        QueryOverlapEdgeRule(config),
        EntityOverlapEdgeRule(config),
        BridgeEdgeRule(config),
    )


def build_graph(task_input: MemoryTaskInput, config: GraphBuildConfig) -> MemoryGraph:
    return GraphBuilder(config).build(task_input)


def build_graphs(task_inputs: list[MemoryTaskInput], config: GraphBuildConfig) -> list[MemoryGraph]:
    return GraphBuilder(config).build_many(task_inputs)

