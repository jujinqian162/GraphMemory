from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.graphs.requests import GraphBuildNode
from graph_memory.text.entities import title_aliases


@dataclass(frozen=True)
class BridgeEdgeRule:
    config: GraphBuildConfig

    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        candidates: list[tuple[float, str, str]] = []
        for left, right in combinations(graph_input.request.nodes, 2):
            if _source_group(left) == _source_group(right):
                continue
            shared_entities = graph_input.entities_by_node_id[left.node_id] & graph_input.entities_by_node_id[right.node_id]
            cross_title_mentions = _title_mention_score(left, right) + _title_mention_score(right, left)
            score = float(len(shared_entities)) + cross_title_mentions
            if score > 0.0:
                candidates.append((score, left.node_id, right.node_id))

        for score, source, target in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))[
            : self.config.max_bridge_edges
        ]:
            accumulator.add(source, target, "bridge", score, directed=False)


def _title_mention_score(left: GraphBuildNode, right: GraphBuildNode) -> float:
    if left.source_ref is None:
        return 0.0
    right_text = right.text.lower()
    return float(sum(1 for alias in title_aliases(left.source_ref) if alias and alias in right_text))


def _source_group(node: GraphBuildNode) -> str | None:
    return node.group_key or node.source_ref
