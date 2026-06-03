from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.graphs.views import induced_retrieved_subgraph
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.methods.graph_rerank.candidates import expanded_candidate_nodes
from graph_memory.retrieval.methods.graph_rerank.components import (
    ScoreContext,
    combine_component_scores_with_breakdown,
    graph_rerank_components,
)
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, RerankResult, ScoreBreakdown
from graph_memory.retrieval.methods.graph_rerank.normalization import normalize_scores
from graph_memory.validation import validate_graph_rerank_config


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
    ranked_nodes, score_breakdown = combine_component_scores_with_breakdown(
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
