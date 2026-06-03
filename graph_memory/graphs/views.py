from __future__ import annotations

from collections import defaultdict

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RetrievedSubgraph


def induced_retrieved_subgraph(graph: MemoryGraph, node_ids: list[str]) -> RetrievedSubgraph:
    selected = set(node_ids)
    return {
        "nodes": list(node_ids),
        "edges": [
            edge
            for edge in graph.get("edges", [])
            if edge.get("source") in selected and edge.get("target") in selected
        ],
    }


def traversal_adjacency(graph: MemoryGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        adjacency[source].add(target)
        if not edge.get("directed", False):
            adjacency[target].add(source)
    return dict(adjacency)


def model_visible_graph(graph: MemoryGraph, enabled_edge_types: frozenset[str]) -> MemoryGraph:
    return {
        **graph,
        "edges": [
            edge
            for edge in graph.get("edges", [])
            if edge.get("edge_type") in enabled_edge_types
        ],
    }

