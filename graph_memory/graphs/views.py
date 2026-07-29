from __future__ import annotations

from collections import defaultdict

from graph_memory.graphs.contracts import EvidenceGraph, GraphEdge


def induced_edges(
    graph: EvidenceGraph, node_ids: list[str]
) -> tuple[GraphEdge, ...]:
    selected = set(node_ids)
    return tuple(
        edge
        for edge in graph.edges
        if edge.source in selected and edge.target in selected
    )


def traversal_adjacency(graph: EvidenceGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        adjacency[edge.source].add(edge.target)
        if not edge.directed:
            adjacency[edge.target].add(edge.source)
    return dict(adjacency)


def model_visible_graph(
    graph: EvidenceGraph, enabled_edge_types: frozenset[str]
) -> EvidenceGraph:
    return graph.model_copy(
        update={
            "edges": tuple(
                edge for edge in graph.edges if edge.edge_type in enabled_edge_types
            )
        }
    )
