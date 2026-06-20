from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.graphs.requests import GraphBuildNode


@dataclass(frozen=True)
class SequentialEdgeRule:
    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        items_by_group: dict[str, list[GraphBuildNode]] = defaultdict(list)
        for node in graph_input.request.nodes:
            group_key = node.group_key or node.source_ref
            if group_key is not None:
                items_by_group[group_key].append(node)

        for grouped_items in items_by_group.values():
            ordered_items = sorted(grouped_items, key=lambda node: node.sequence_index or 0)
            for left, right in zip(ordered_items, ordered_items[1:]):
                if left.sequence_index is None or right.sequence_index is None:
                    continue
                if right.sequence_index - left.sequence_index == 1:
                    accumulator.add(left.node_id, right.node_id, "sequential", 1.0, directed=False)
