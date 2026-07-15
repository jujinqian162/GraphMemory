from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ProvenanceEdgeType,
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


def enumerate_provenance_paths(
    request: ExecutionProvenanceRankingRequest,
    seed_ids: tuple[str, ...] | list[str],
    config: ExecutionProvenanceConfig,
) -> list[ProvenancePath]:
    adjacency: dict[str, list[tuple[str, ExecutionProvenanceEdge]]] = defaultdict(list)
    for edge in request.graph.edges:
        if edge.edge_type not in DEPENDENCY_EDGE_TYPES:
            continue
        adjacency[edge.source].append((edge.target, edge))
        adjacency[edge.target].append((edge.source, edge))
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda pair: (_edge_key(pair[1]), pair[0]))

    candidate_ids = {candidate.item_id for candidate in request.candidates}
    paths: list[ProvenancePath] = []
    seen_paths: set[tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]] = set()
    for seed_id in seed_ids:
        queue = deque([ProvenancePath((seed_id,), ())])
        expansion_count = 0
        while queue and expansion_count < config.max_path_expansions:
            current = queue.popleft()
            if len(current.edges) >= config.max_hops:
                continue
            for neighbor, edge in adjacency.get(current.node_ids[-1], []):
                if neighbor in current.node_ids:
                    continue
                next_path = ProvenancePath(
                    (*current.node_ids, neighbor),
                    (*current.edges, edge),
                )
                expansion_count += 1
                if neighbor in candidate_ids and neighbor != seed_id:
                    key = (
                        next_path.node_ids,
                        tuple(_edge_key(item) for item in next_path.edges),
                    )
                    if key not in seen_paths:
                        seen_paths.add(key)
                        paths.append(next_path)
                if (
                    len(next_path.edges) < config.max_hops
                    and expansion_count < config.max_path_expansions
                ):
                    queue.append(next_path)
    return paths


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
    config: ExecutionProvenanceConfig,
) -> ProvenancePathScore:
    edge_types = {edge.edge_type for edge in path.edges}
    semantic = max(
        (semantic_scores.get(node_id, 0.0) for node_id in path.node_ids),
        default=0.0,
    )
    feeds_edges = [
        edge for edge in path.edges if edge.edge_type is ProvenanceEdgeType.FEEDS
    ]
    binding = (
        sum(edge.binding is not None for edge in feeds_edges) / len(feeds_edges)
        if feeds_edges
        else 0.0
    )
    completeness = float(
        ProvenanceEdgeType.RETURNS in edge_types
        and ProvenanceEdgeType.FEEDS in edge_types
    )
    grounding = float(
        bool({ProvenanceEdgeType.GROUNDS, ProvenanceEdgeType.SUPPORTS} & edge_types)
    )
    length_penalty = config.hop_penalty * len(path.edges)
    invalidation_penalty = config.invalidation_penalty * float(
        any(node_id in invalidated_node_ids for node_id in path.node_ids)
    )
    total = (
        config.semantic_weight * semantic
        + config.binding_weight * binding
        + config.dependency_weight * completeness
        + config.grounding_weight * grounding
        - length_penalty
        - invalidation_penalty
    )
    return ProvenancePathScore(
        semantic_relevance=semantic,
        binding_consistency=binding,
        provenance_completeness=completeness,
        explicit_grounding=grounding,
        path_length_penalty=length_penalty,
        invalidation_penalty=invalidation_penalty,
        total=total,
    )


def _edge_key(edge: ExecutionProvenanceEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.edge_type.value)


__all__ = [
    "DEPENDENCY_EDGE_TYPES",
    "ProvenancePath",
    "ProvenancePathScore",
    "enumerate_provenance_paths",
    "invalidated_node_ids",
    "score_provenance_path",
]
