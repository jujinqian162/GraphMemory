from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator


@dataclass(frozen=True)
class EntityOverlapEdgeRule:
    config: GraphBuildConfig

    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        candidates: list[tuple[float, str, str]] = []
        for left, right in combinations(graph_input.request.nodes, 2):
            score = float(len(graph_input.entities_by_node_id[left.node_id] & graph_input.entities_by_node_id[right.node_id]))
            if score > 0.0:
                candidates.append((score, left.node_id, right.node_id))

        neighbor_counts: dict[str, int] = defaultdict(int)
        for score, source, target in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2])):
            if (
                neighbor_counts[source] >= self.config.max_entity_neighbors
                or neighbor_counts[target] >= self.config.max_entity_neighbors
            ):
                continue
            accumulator.add(source, target, "entity_overlap", score, directed=False)
            neighbor_counts[source] += 1
            neighbor_counts[target] += 1
