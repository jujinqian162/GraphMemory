from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

from graph_memory.contracts.common import NodeId

DependencyEdge = tuple[NodeId, NodeId]


def edge_recall_at(retrieved_subgraph: Mapping[str, object], gold_dependency_edges: set[DependencyEdge]) -> float:
    if not gold_dependency_edges:
        return 0.0
    retrieved_nodes = _retrieved_nodes(retrieved_subgraph)
    retrieved_edges = _retrieved_edges(retrieved_subgraph)
    covered = 0
    for source, target in gold_dependency_edges:
        if source not in retrieved_nodes or target not in retrieved_nodes:
            continue
        if _has_direct_visible_edge(retrieved_edges, source, target):
            covered += 1
    return covered / len(gold_dependency_edges)


def path_recall_at(retrieved_subgraph: Mapping[str, object], gold_dependency_edges: set[DependencyEdge]) -> float:
    if not gold_dependency_edges:
        return 0.0
    retrieved_nodes = _retrieved_nodes(retrieved_subgraph)
    gold_nodes = {node_id for edge in gold_dependency_edges for node_id in edge}
    if not gold_nodes.issubset(retrieved_nodes):
        return 0.0
    adjacency = _traversal_adjacency(_retrieved_edges(retrieved_subgraph), retrieved_nodes)
    for source, target in gold_dependency_edges:
        if target not in _reachable_from(source, adjacency):
            return 0.0
    return 1.0


def _retrieved_nodes(retrieved_subgraph: Mapping[str, object]) -> set[NodeId]:
    nodes = retrieved_subgraph.get("nodes", [])
    if not isinstance(nodes, list):
        return set()
    return {node for node in nodes if isinstance(node, str)}


def _retrieved_edges(retrieved_subgraph: Mapping[str, object]) -> list[Mapping[str, object]]:
    edges = retrieved_subgraph.get("edges", [])
    if not isinstance(edges, list):
        return []
    return [edge for edge in edges if isinstance(edge, Mapping)]


def _has_direct_visible_edge(edges: Iterable[Mapping[str, object]], source: NodeId, target: NodeId) -> bool:
    for edge in edges:
        edge_source = edge.get("source")
        edge_target = edge.get("target")
        if edge_source == source and edge_target == target:
            return True
        if not edge.get("directed", False) and edge_source == target and edge_target == source:
            return True
    return False


def _traversal_adjacency(
    edges: Iterable[Mapping[str, object]],
    allowed_nodes: set[NodeId],
) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = defaultdict(set)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source not in allowed_nodes or target not in allowed_nodes:
            continue
        adjacency[source].add(target)
        if not edge.get("directed", False):
            adjacency[target].add(source)
    return dict(adjacency)


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


__all__ = ["edge_recall_at", "path_recall_at"]
