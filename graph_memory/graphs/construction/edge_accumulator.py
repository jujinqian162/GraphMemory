from __future__ import annotations

from dataclasses import dataclass, field

from graph_memory.contracts.common import EdgeType
from graph_memory.contracts.graphs import GraphEdge


@dataclass
class EdgeAccumulator:
    edges: list[GraphEdge] = field(default_factory=list)
    seen_edge_keys: set[tuple[str, str, str]] = field(default_factory=set)

    def add(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float,
        *,
        directed: bool,
    ) -> None:
        edge_key = self._edge_key(source, target, edge_type, directed=directed)
        if edge_key in self.seen_edge_keys:
            return
        self.seen_edge_keys.add(edge_key)
        self.edges.append(
            {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "weight": weight,
                "directed": directed,
            }
        )

    @staticmethod
    def _edge_key(source: str, target: str, edge_type: str, *, directed: bool) -> tuple[str, str, str]:
        if directed:
            return source, target, edge_type
        left, right = sorted([source, target])
        return left, right, edge_type
