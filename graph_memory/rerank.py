from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from graph_memory.types import (
    GraphRerankConfig,
    MemoryGraph,
    RankedNode,
    RerankResult,
    RetrievedSubgraph,
    ScoreBreakdown,
    ScoreComponents,
)
from graph_memory.validation import validate_graph_rerank_config

NormalizationMode = Literal["none", "minmax"]
ComponentName = Literal["initial", "query", "neighbor", "bridge", "path"]


class NodeScoreComponent(Protocol):
    @property
    def component_name(self) -> ComponentName:
        ...

    @property
    def weight(self) -> float:
        ...

    @property
    def normalization(self) -> NormalizationMode:
        ...

    def scores(self, context: ScoreContext) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class ScoreContext:
    initial_scores: dict[str, float]
    normalized_initial: dict[str, float]
    graph: MemoryGraph | None = None
    graph_config: GraphRerankConfig | None = None
    candidate_nodes: set[str] | None = None


@dataclass(frozen=True)
class InitialScoreComponent:
    weight: float
    normalization: NormalizationMode
    component_name: ComponentName = "initial"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        return context.initial_scores


@dataclass(frozen=True)
class QueryOverlapScoreComponent:
    weight: float
    normalization: NormalizationMode = "minmax"
    component_name: ComponentName = "query"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        if context.graph is None:
            return {}
        return _filter_candidate_scores(query_overlap_scores(context.graph), context.candidate_nodes)


@dataclass(frozen=True)
class NeighborPropagationScoreComponent:
    weight: float
    normalization: NormalizationMode = "minmax"
    component_name: ComponentName = "neighbor"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        if context.graph is None or context.graph_config is None:
            return {}
        scores = neighbor_propagation_scores(context.normalized_initial, context.graph, context.graph_config)
        return _filter_candidate_scores(scores, context.candidate_nodes)


@dataclass(frozen=True)
class BridgeScoreComponent:
    weight: float
    normalization: NormalizationMode = "minmax"
    component_name: ComponentName = "bridge"

    def scores(self, context: ScoreContext) -> dict[str, float]:
        if context.graph is None or context.graph_config is None:
            return {}
        scores = bridge_edge_scores(context.normalized_initial, context.graph, context.graph_config)
        return _filter_candidate_scores(scores, context.candidate_nodes)


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
    return rank_graph_from_initial_scores(
        initial_scores,
        graph,
        config,
        top_k=len(initial_scores),
    ).ranked_nodes


def graph_rerank_with_breakdown(
    initial_scores: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
) -> tuple[list[RankedNode], ScoreBreakdown]:
    result = rank_graph_from_initial_scores(
        initial_scores,
        graph,
        config,
        top_k=len(initial_scores),
        include_score_breakdown=True,
    )
    return result.ranked_nodes, result.score_breakdown or {}


def rank_graph_from_initial_scores(
    initial_scores: dict[str, float],
    graph: MemoryGraph,
    config: GraphRerankConfig,
    *,
    top_k: int,
    include_score_breakdown: bool = False,
) -> RerankResult:
    validate_graph_rerank_config(config)
    normalized_initial = normalize_scores(initial_scores)
    candidate_nodes = expanded_candidate_nodes(normalized_initial, graph, config)
    context = ScoreContext(
        initial_scores=initial_scores,
        normalized_initial=normalized_initial,
        graph=graph,
        graph_config=config,
        candidate_nodes=candidate_nodes,
    )
    ranked_nodes, score_breakdown = _combine_component_scores_with_breakdown(
        initial_scores,
        context,
        graph_rerank_components(config),
        include_breakdown=include_score_breakdown,
    )
    top_node_ids = [ranked_node.node_id for ranked_node in ranked_nodes[:top_k]]
    return RerankResult(
        ranked_nodes=ranked_nodes,
        retrieved_subgraph=induced_retrieved_subgraph(graph, top_node_ids),
        score_breakdown=score_breakdown,
    )


def graph_rerank_components(config: GraphRerankConfig) -> list[NodeScoreComponent]:
    return [
        InitialScoreComponent(weight=config.lambda_init, normalization="minmax"),
        QueryOverlapScoreComponent(weight=config.lambda_query),
        NeighborPropagationScoreComponent(weight=config.lambda_neighbor),
        BridgeScoreComponent(weight=config.lambda_bridge),
    ]


def combine_component_scores(
    node_scores: dict[str, float],
    context: ScoreContext,
    components: Sequence[NodeScoreComponent],
) -> list[RankedNode]:
    ranked_nodes, _ = _combine_component_scores_with_breakdown(
        node_scores,
        context,
        components,
        include_breakdown=False,
    )
    return ranked_nodes


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
        weight = float(edge.get("weight", 0.0)) * config.neighbor_type_weights.get(str(edge.get("edge_type")), 0.0)
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
        weight = float(edge.get("weight", 0.0)) * config.neighbor_type_weights.get("bridge", 1.0)
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


def _normalize_component_scores(
    scores: dict[str, float],
    mode: NormalizationMode,
    node_ids: set[str],
) -> dict[str, float]:
    if mode == "none":
        return scores
    return normalize_component_scores(scores, node_ids)


def _combine_component_scores_with_breakdown(
    node_scores: dict[str, float],
    context: ScoreContext,
    components: Sequence[NodeScoreComponent],
    *,
    include_breakdown: bool,
) -> tuple[list[RankedNode], ScoreBreakdown | None]:
    combined_scores = {node_id: 0.0 for node_id in node_scores}
    component_values: dict[str, dict[ComponentName, float]] = {
        node_id: {} for node_id in node_scores
    }
    for component in components:
        component_scores = _normalize_component_scores(
            component.scores(context),
            component.normalization,
            set(combined_scores),
        )
        for node_id in combined_scores:
            contribution = component.weight * component_scores.get(node_id, 0.0)
            combined_scores[node_id] += contribution
            if include_breakdown:
                component_values[node_id][component.component_name] = contribution
    ranked_nodes = [
        RankedNode(node_id=node_id, score=score)
        for node_id, score in combined_scores.items()
    ]
    sorted_nodes = sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))
    if not include_breakdown:
        return sorted_nodes, None

    score_breakdown: ScoreBreakdown = {}
    for node_id, final_score in combined_scores.items():
        values = component_values[node_id]
        score_breakdown[node_id] = ScoreComponents(
            initial=values.get("initial", 0.0),
            query=values.get("query", 0.0),
            neighbor=values.get("neighbor", 0.0),
            bridge=values.get("bridge", 0.0),
            path=values.get("path", 0.0),
            final=final_score,
        )
    return sorted_nodes, score_breakdown


def _filter_candidate_scores(scores: dict[str, float], candidate_nodes: set[str] | None) -> dict[str, float]:
    if candidate_nodes is None:
        return scores
    return {node_id: score for node_id, score in scores.items() if node_id in candidate_nodes}
