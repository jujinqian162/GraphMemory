from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceNode,
    ProvenanceEdgeType,
    binding_matches_endpoints,
)
from graph_memory.retrieval.methods.execution_provenance.config import (
    ExecutionProvenanceConfig,
)
from graph_memory.retrieval.requests import ExecutionProvenanceRankingRequest

DEPENDENCY_EDGE_TYPES = frozenset(
    {
        ProvenanceEdgeType.RETURNS,
        ProvenanceEdgeType.FEEDS,
        ProvenanceEdgeType.GROUNDS,
        ProvenanceEdgeType.SUPPORTS,
        ProvenanceEdgeType.DEPENDS_ON,
    }
)
REVISION_EDGE_TYPES = frozenset(
    {ProvenanceEdgeType.INVALIDATES, ProvenanceEdgeType.SUPERSEDES}
)
INVALID_LIFECYCLE_STATES = frozenset(
    {"invalid", "invalidated", "superseded", "obsolete"}
)


@dataclass(frozen=True)
class ProvenancePath:
    node_ids: tuple[str, ...]
    edges: tuple[ExecutionProvenanceEdge, ...]


@dataclass(frozen=True)
class ProvenancePathScore:
    semantic_relevance: float
    binding_consistency: float
    provenance_completeness: float
    explicit_grounding: float
    path_length_penalty: float
    invalidation_penalty: float
    total: float


def beam_search_provenance_paths(
    request: ExecutionProvenanceRankingRequest,
    seed_ids: tuple[str, ...] | list[str],
    *,
    semantic_scores: dict[str, float],
    invalidated_node_ids: frozenset[str],
    config: ExecutionProvenanceConfig,
) -> list[ProvenancePath]:
    adjacency = _directed_adjacency(request)
    node_by_id = {node.node_id: node for node in request.graph.nodes}
    candidate_ids = {candidate.item_id for candidate in request.candidates}
    beam = [ProvenancePath((seed_id,), ()) for seed_id in seed_ids]
    completed: dict[
        tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]], ProvenancePath
    ] = {}
    expansion_count = 0
    for _hop in range(config.max_hops):
        expanded: list[tuple[ProvenancePathScore, ProvenancePath]] = []
        for path in beam:
            for neighbor, edge in adjacency.get(path.node_ids[-1], ()):
                if expansion_count >= config.max_path_expansions:
                    break
                if neighbor in path.node_ids:
                    continue
                next_path = ProvenancePath(
                    node_ids=(*path.node_ids, neighbor),
                    edges=(*path.edges, edge),
                )
                expansion_count += 1
                score = score_provenance_path(
                    next_path,
                    semantic_scores=semantic_scores,
                    invalidated_node_ids=invalidated_node_ids,
                    node_by_id=node_by_id,
                    config=config,
                )
                expanded.append((score, next_path))
                if neighbor in candidate_ids and neighbor != next_path.node_ids[0]:
                    completed[_path_key(next_path)] = next_path
            if expansion_count >= config.max_path_expansions:
                break
        if not expanded:
            break
        expanded.sort(key=_scored_path_sort_key)
        beam = [path for _score, path in expanded[: config.beam_width]]
        if expansion_count >= config.max_path_expansions:
            break
    return sorted(
        completed.values(),
        key=lambda path: (
            len(path.edges),
            path.node_ids,
            tuple(_edge_key(edge) for edge in path.edges),
        ),
    )


def enumerate_provenance_paths(
    request: ExecutionProvenanceRankingRequest,
    seed_ids: tuple[str, ...] | list[str],
    config: ExecutionProvenanceConfig,
) -> list[ProvenancePath]:
    """Compatibility entrypoint backed by the bounded directed beam search."""
    return beam_search_provenance_paths(
        request,
        seed_ids,
        semantic_scores={},
        invalidated_node_ids=invalidated_node_ids(request),
        config=config,
    )


def invalidated_node_ids(
    request: ExecutionProvenanceRankingRequest,
) -> frozenset[str]:
    invalidated = {
        edge.target
        for edge in request.graph.edges
        if edge.edge_type in REVISION_EDGE_TYPES
    }
    for node in request.graph.nodes:
        lifecycle_state = node.metadata.get("lifecycle_state")
        valid = node.metadata.get("valid")
        if (
            isinstance(lifecycle_state, str)
            and lifecycle_state.casefold() in INVALID_LIFECYCLE_STATES
        ) or valid is False:
            invalidated.add(node.node_id)
    return frozenset(invalidated)


def score_provenance_path(
    path: ProvenancePath,
    *,
    semantic_scores: dict[str, float],
    invalidated_node_ids: frozenset[str],
    node_by_id: Mapping[str, ExecutionProvenanceNode],
    config: ExecutionProvenanceConfig,
) -> ProvenancePathScore:
    available_semantics = [
        semantic_scores[node_id]
        for node_id in path.node_ids
        if node_id in semantic_scores
    ]
    endpoint = semantic_scores.get(path.node_ids[-1], 0.0)
    mean_semantic = (
        sum(available_semantics) / len(available_semantics)
        if available_semantics
        else 0.0
    )
    bottleneck = min(available_semantics, default=0.0)
    semantic = 0.5 * endpoint + 0.3 * mean_semantic + 0.2 * bottleneck
    feeds_edges = [
        edge for edge in path.edges if edge.edge_type is ProvenanceEdgeType.FEEDS
    ]
    binding = (
        sum(binding_matches_endpoints(edge, node_by_id) for edge in feeds_edges)
        / len(feeds_edges)
        if feeds_edges
        else 0.0
    )
    edge_types = {edge.edge_type for edge in path.edges}
    completeness = 0.5 * float(ProvenanceEdgeType.FEEDS in edge_types) + 0.5 * float(
        ProvenanceEdgeType.RETURNS in edge_types
    )
    grounding = float(
        bool({ProvenanceEdgeType.GROUNDS, ProvenanceEdgeType.SUPPORTS} & edge_types)
    )
    length_penalty = config.hop_penalty * max(0, len(path.edges) - 2)
    lifecycle_penalty = config.invalidation_penalty * float(
        any(node_id in invalidated_node_ids for node_id in path.node_ids)
    )
    total = (
        config.semantic_weight * semantic
        + config.binding_weight * binding
        + config.dependency_weight * completeness
        + config.grounding_weight * grounding
        - length_penalty
        - lifecycle_penalty
    )
    return ProvenancePathScore(
        semantic_relevance=semantic,
        binding_consistency=binding,
        provenance_completeness=completeness,
        explicit_grounding=grounding,
        path_length_penalty=length_penalty,
        invalidation_penalty=lifecycle_penalty,
        total=total,
    )


def _directed_adjacency(
    request: ExecutionProvenanceRankingRequest,
) -> dict[str, tuple[tuple[str, ExecutionProvenanceEdge], ...]]:
    mutable: dict[str, list[tuple[str, ExecutionProvenanceEdge]]] = defaultdict(list)
    for edge in request.graph.edges:
        if edge.edge_type in DEPENDENCY_EDGE_TYPES:
            mutable[edge.source].append((edge.target, edge))
    return {
        source: tuple(sorted(neighbors, key=lambda pair: (_edge_key(pair[1]), pair[0])))
        for source, neighbors in mutable.items()
    }


def _scored_path_sort_key(
    record: tuple[ProvenancePathScore, ProvenancePath],
) -> tuple[float, int, tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    score, path = record
    return (
        -score.total,
        len(path.edges),
        path.node_ids,
        tuple(_edge_key(edge) for edge in path.edges),
    )


def _path_key(
    path: ProvenancePath,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    return path.node_ids, tuple(_edge_key(edge) for edge in path.edges)


def _edge_key(edge: ExecutionProvenanceEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.edge_type.value)


__all__ = [
    "DEPENDENCY_EDGE_TYPES",
    "ProvenancePath",
    "ProvenancePathScore",
    "beam_search_provenance_paths",
    "enumerate_provenance_paths",
    "invalidated_node_ids",
    "score_provenance_path",
]
