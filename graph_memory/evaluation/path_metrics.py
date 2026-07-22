from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.graphs import GraphEdge
from graph_memory.contracts.ranking import RetrievedSubgraph

DependencyEdge = tuple[NodeId, NodeId]


def edge_recall_at(
    retrieved_subgraph: RetrievedSubgraph,
    gold_dependency_edges: set[DependencyEdge],
) -> float:
    if not gold_dependency_edges:
        return 0.0
    retrieved_nodes = set(retrieved_subgraph["nodes"])
    retrieved_edges = retrieved_subgraph["edges"]
    covered = 0
    for source, target in gold_dependency_edges:
        if source not in retrieved_nodes or target not in retrieved_nodes:
            continue
        if _has_direct_visible_edge(retrieved_edges, source, target):
            covered += 1
    return covered / len(gold_dependency_edges)


def path_recall_at(
    retrieved_subgraph: RetrievedSubgraph,
    gold_dependency_edges: set[DependencyEdge],
) -> float:
    if not gold_dependency_edges:
        return 0.0
    retrieved_nodes = set(retrieved_subgraph["nodes"])
    gold_nodes = {node_id for edge in gold_dependency_edges for node_id in edge}
    if not gold_nodes.issubset(retrieved_nodes):
        return 0.0
    adjacency = _traversal_adjacency(
        retrieved_subgraph["edges"],
        retrieved_nodes,
    )
    for source, target in gold_dependency_edges:
        if target not in _reachable_from(source, adjacency):
            return 0.0
    return 1.0


def _has_direct_visible_edge(
    edges: Iterable[GraphEdge],
    source: NodeId,
    target: NodeId,
) -> bool:
    for edge in edges:
        edge_source = edge["source"]
        edge_target = edge["target"]
        if edge_source == source and edge_target == target:
            return True
        if not edge["directed"] and edge_source == target and edge_target == source:
            return True
    return False


def _traversal_adjacency(
    edges: Iterable[GraphEdge],
    allowed_nodes: set[NodeId],
) -> dict[NodeId, set[NodeId]]:
    adjacency: dict[NodeId, set[NodeId]] = defaultdict(set)
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source not in allowed_nodes or target not in allowed_nodes:
            continue
        adjacency[source].add(target)
        if not edge["directed"]:
            adjacency[target].add(source)
    return dict(adjacency)


def _reachable_from(
    start_node: NodeId,
    adjacency: dict[NodeId, set[NodeId]],
) -> set[NodeId]:
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
