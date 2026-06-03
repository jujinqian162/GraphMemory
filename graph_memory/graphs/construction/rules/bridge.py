from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from graph_memory.contracts.tasks import MemoryItem
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator
from graph_memory.text.entities import title_aliases


@dataclass(frozen=True)
class BridgeEdgeRule:
    config: GraphBuildConfig

    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        candidates: list[tuple[float, str, str]] = []
        for left, right in combinations(graph_input.task_input["memory_items"], 2):
            if left["source"] == right["source"]:
                continue
            shared_entities = graph_input.entities_by_node_id[left["id"]] & graph_input.entities_by_node_id[right["id"]]
            cross_title_mentions = _title_mention_score(left, right) + _title_mention_score(right, left)
            score = float(len(shared_entities)) + cross_title_mentions
            if score > 0.0:
                candidates.append((score, left["id"], right["id"]))

        for score, source, target in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))[
            : self.config.max_bridge_edges
        ]:
            accumulator.add(source, target, "bridge", score, directed=False)


def _title_mention_score(left: MemoryItem, right: MemoryItem) -> float:
    right_text = right["text"].lower()
    return float(sum(1 for alias in title_aliases(left["source"]) if alias and alias in right_text))

