from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.graphs import GraphEdge, MemoryGraph
from graph_memory.evaluation.metrics import require_gold_nodes


@dataclass(frozen=True)
class GraphConnectivity:
    directed_adjacency: dict[NodeId, set[NodeId]]
    undirected_adjacency: dict[NodeId, set[NodeId]]

    @classmethod
    def from_graph(cls, graph: MemoryGraph, allowed_nodes: set[NodeId]) -> "GraphConnectivity":
        edges = graph.get("edges", [])
        return cls(
            directed_adjacency=_directed_adjacency(edges, allowed_nodes),
            undirected_adjacency=_undirected_adjacency(edges, allowed_nodes),
        )

    def directed_reachable(self, start_node: NodeId) -> set[NodeId]:
        return _reachable_from(start_node, self.directed_adjacency)

    def undirected_reachable(self, start_node: NodeId) -> set[NodeId]:
        return _reachable_from(start_node, self.undirected_adjacency)


def connected_evidence_at(ranked_nodes: list[NodeId], gold_nodes: set[NodeId], graph: MemoryGraph, k: int) -> float:
    require_gold_nodes(gold_nodes)
    selected = set(ranked_nodes[:k])
    if not gold_nodes.issubset(selected):
        return 0.0
    if len(gold_nodes) == 1:
        return 1.0
    connectivity = GraphConnectivity.from_graph(graph, selected)
    first_gold = next(iter(gold_nodes))
    reachable = connectivity.undirected_reachable(first_gold)
    return 1.0 if gold_nodes.issubset(reachable) else 0.0


def query_evidence_connectivity_at(
    ranked_nodes: list[NodeId],
    gold_nodes: set[NodeId],
    graph: MemoryGraph,
    k: int,
) -> float:
    require_gold_nodes(gold_nodes)
    selected = set(ranked_nodes[:k])
    if not gold_nodes.issubset(selected):
        return 0.0
    allowed_nodes = selected | {"q"}
    connectivity = GraphConnectivity.from_graph(graph, allowed_nodes)
    reachable = connectivity.directed_reachable("q")
    return 1.0 if gold_nodes.issubset(reachable) else 0.0


def _undirected_adjacency(edges: Iterable[GraphEdge], allowed_nodes: set[NodeId]) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in allowed_nodes and target in allowed_nodes:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def _directed_adjacency(edges: Iterable[GraphEdge], allowed_nodes: set[NodeId]) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in allowed_nodes or target not in allowed_nodes:
            continue
        adjacency[source].add(target)
        if not edge.get("directed", False):
            adjacency[target].add(source)
    return adjacency


def _reachable_from(start_node: NodeId, adjacency: dict[NodeId, set[NodeId]]) -> set[NodeId]:
    seen = {start_node}
    queue: deque[NodeId] = deque([start_node])
    while queue:
        node_id = queue.popleft()
        for neighbor in adjacency.get(node_id, set()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


__all__ = ["GraphConnectivity", "connected_evidence_at", "query_evidence_connectivity_at"]
