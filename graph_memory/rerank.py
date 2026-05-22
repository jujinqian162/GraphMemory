from __future__ import annotations

from collections import defaultdict, deque

from graph_memory.types import GraphRerankConfig, MemoryGraph, RankedNode, RetrievedSubgraph, ScoreBreakdown, ScoreComponents
from graph_memory.validation import validate_graph_rerank_config


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {node_id: 0.0 for node_id in scores}
    return {node_id: (score - min_score) / (max_score - min_score) for node_id, score in scores.items()}


def normalize_component_scores(scores: dict[str, float], node_ids: set[str]) -> dict[str, float]:
    """Normalize one score component against every memory node in a task."""

    zero_filled_scores = {node_id: scores.get(node_id, 0.0) for node_id in node_ids}
    return normalize_scores(zero_filled_scores)


def graph_rerank(initial_scores: dict[str, float], graph: MemoryGraph, config: GraphRerankConfig) -> list[RankedNode]:
    ranked_nodes, _ = graph_rerank_with_breakdown(initial_scores, graph, config)
    return ranked_nodes


def graph_rerank_with_breakdown(
    initial_scores: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
) -> tuple[list[RankedNode], ScoreBreakdown]:
    validate_graph_rerank_config(config)
    normalized_initial = normalize_scores(initial_scores)
    candidate_nodes = expanded_candidate_nodes(normalized_initial, graph, config)
    memory_node_ids = set(initial_scores)
    query_scores = normalize_component_scores(
        _filter_candidate_scores(query_overlap_scores(graph), candidate_nodes),
        memory_node_ids,
    )
    neighbor_scores = normalize_component_scores(
        _filter_candidate_scores(neighbor_propagation_scores(normalized_initial, graph, config), candidate_nodes),
        memory_node_ids,
    )
    bridge_scores = normalize_component_scores(
        _filter_candidate_scores(bridge_edge_scores(normalized_initial, graph, config), candidate_nodes),
        memory_node_ids,
    )

    score_breakdown: ScoreBreakdown = {}
    reranked_nodes: list[RankedNode] = []
    for node_id in initial_scores:
        initial_component = config.lambda_init * normalized_initial[node_id]
        query_component = config.lambda_query * query_scores.get(node_id, 0.0) if node_id in candidate_nodes else 0.0
        neighbor_component = config.lambda_neighbor * neighbor_scores.get(node_id, 0.0) if node_id in candidate_nodes else 0.0
        bridge_component = config.lambda_bridge * bridge_scores.get(node_id, 0.0) if node_id in candidate_nodes else 0.0
        path_component = 0.0
        final_score = initial_component + query_component + neighbor_component + bridge_component + path_component
        score_breakdown[node_id] = ScoreComponents(
            initial=initial_component,
            query=query_component,
            neighbor=neighbor_component,
            bridge=bridge_component,
            path=path_component,
            final=final_score,
        )
        reranked_nodes.append(RankedNode(node_id=node_id, score=final_score))

    return sorted(reranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id)), score_breakdown


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


def expanded_candidate_nodes(
    normalized_initial: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
) -> set[str]:
    seeds = [
        node_id
        for node_id, _ in sorted(normalized_initial.items(), key=lambda item: (-item[1], item[0]))[: config.seed_top_s]
    ]
    adjacency = _traversal_adjacency(graph)
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


def query_overlap_scores(graph: MemoryGraph) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for edge in graph.get("edges", []):
        if edge.get("source") == "q" and edge.get("edge_type") == "query_overlap":
            scores[str(edge["target"])] += float(edge["weight"])
    return dict(scores)


def neighbor_propagation_scores(
    normalized_initial: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    normalizers: dict[str, float] = defaultdict(float)
    for edge in graph.get("edges", []):
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source == "q" or target == "q":
            continue
        weight = float(edge.get("weight", 0.0)) * config.type_weights.get(str(edge.get("edge_type")), 0.0)
        if weight <= 0.0:
            continue
        if source in normalized_initial and target in normalized_initial:
            scores[target] += normalized_initial[source] * weight
            normalizers[target] += weight
            if not edge.get("directed", False):
                scores[source] += normalized_initial[target] * weight
                normalizers[source] += weight
    return {
        node_id: score / normalizers[node_id]
        for node_id, score in scores.items()
        if normalizers[node_id] > 0.0
    }


def bridge_edge_scores(
    normalized_initial: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for edge in graph.get("edges", []):
        if edge.get("edge_type") != "bridge":
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in normalized_initial or target not in normalized_initial:
            continue
        weight = float(edge.get("weight", 0.0)) * config.type_weights.get("bridge", 1.0)
        scores[target] += normalized_initial[source] * weight
        if not edge.get("directed", False):
            scores[source] += normalized_initial[target] * weight
    return dict(scores)


def _traversal_adjacency(graph: MemoryGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        adjacency[source].add(target)
        if not edge.get("directed", False):
            adjacency[target].add(source)
    return adjacency


def _filter_candidate_scores(scores: dict[str, float], candidate_nodes: set[str]) -> dict[str, float]:
    return {node_id: score for node_id, score in scores.items() if node_id in candidate_nodes}
