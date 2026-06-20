from __future__ import annotations

from collections import Counter

from graph_memory.contracts.common import JsonObject
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.observability import GraphStatistics


def graph_statistics(
    graphs: list[MemoryGraph], *, split: str | None = None, graph_config: JsonObject | None = None
) -> GraphStatistics:
    edge_counts: Counter[str] = Counter()
    total_nodes = 0
    total_edges = 0
    isolated_memory_nodes = 0

    for graph in graphs:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        total_nodes += len(nodes)
        total_edges += len(edges)
        for edge in edges:
            edge_counts[str(edge.get("edge_type"))] += 1

        incident_node_ids: set[str] = set()
        for edge in edges:
            incident_node_ids.add(str(edge.get("source")))
            incident_node_ids.add(str(edge.get("target")))
        for node in nodes:
            if node.get("id") != "q" and node.get("id") not in incident_node_ids:
                isolated_memory_nodes += 1

    num_graphs = len(graphs)
    stats: GraphStatistics = {
        "num_graphs": num_graphs,
        "avg_nodes": total_nodes / num_graphs if num_graphs else 0.0,
        "avg_edges": total_edges / num_graphs if num_graphs else 0.0,
        "edge_counts_by_type": dict(sorted(edge_counts.items())),
        "isolated_memory_nodes": isolated_memory_nodes,
    }
    if split is not None:
        stats["split"] = split
    if graph_config is not None:
        stats["graph_config"] = graph_config
    return stats
