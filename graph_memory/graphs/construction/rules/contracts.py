from __future__ import annotations

from typing import Protocol

from graph_memory.graphs.construction.context import PreparedGraphInput
from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator


class GraphEdgeRule(Protocol):
    def add_edges(self, graph_input: PreparedGraphInput, accumulator: EdgeAccumulator) -> None:
        ...

