from __future__ import annotations

from collections import deque

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.graphs.views import traversal_adjacency
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig


def expanded_candidate_nodes(
    normalized_initial: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
) -> set[str]:
    seeds = [
        node_id
        for node_id, _ in sorted(normalized_initial.items(), key=lambda item: (-item[1], item[0]))[: config.seed_top_s]
    ]
    adjacency = traversal_adjacency(graph)
    candidates = set(seeds)
    queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
    while queue:
        node_id, depth = queue.popleft()
        if depth >= config.max_hops:
            continue
        for neighbor in adjacency.get(node_id, set()):
            if neighbor == "q" or neighbor in candidates:
                continue
            candidates.add(neighbor)
            queue.append((neighbor, depth + 1))
    return candidates
