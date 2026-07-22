from __future__ import annotations

from collections import defaultdict

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.ranking import RetrievedSubgraph


def induced_retrieved_subgraph(
    graph: EvidenceGraph, node_ids: list[str]
) -> RetrievedSubgraph:
    selected = set(node_ids)
    return {
        "nodes": list(node_ids),
        "edges": [
            edge
            for edge in graph["edges"]
            if edge["source"] in selected and edge["target"] in selected
        ],
    }


def traversal_adjacency(graph: EvidenceGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph["edges"]:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].add(target)
        if not edge["directed"]:
            adjacency[target].add(source)
    return dict(adjacency)


def model_visible_graph(
    graph: EvidenceGraph, enabled_edge_types: frozenset[str]
) -> EvidenceGraph:
    return {
        **graph,
        "edges": [
            edge for edge in graph["edges"] if edge["edge_type"] in enabled_edge_types
        ],
    }
