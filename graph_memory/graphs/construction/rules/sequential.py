from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graph_memory.contracts.tasks import MemoryItem
from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator


@dataclass(frozen=True)
class SequentialEdgeRule:
    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        items_by_source: dict[str, list[MemoryItem]] = defaultdict(list)
        for item in graph_input.task_input["memory_items"]:
            items_by_source[item["source"]].append(item)

        for source_items in items_by_source.values():
            ordered_items = sorted(source_items, key=lambda item: item["sentence_id"])
            for left, right in zip(ordered_items, ordered_items[1:]):
                if right["sentence_id"] - left["sentence_id"] == 1:
                    accumulator.add(left["id"], right["id"], "sequential", 1.0, directed=False)

