from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
    path_confidence: float
    binding_valid: bool
    completeness_valid: bool
    lifecycle_valid: bool
    valid: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class ProvenancePathEvaluation:
    anchor_id: str
    partner_id: str
    path: ProvenancePath
    score: ProvenancePathScore

    def __hash__(self) -> int:
        return hash(
            (
                self.anchor_id,
                self.partner_id,
                self.path.node_ids,
                tuple(_edge_key(edge) for edge in self.path.edges),
            )
        )


def search_provenance_paths(
    request: ExecutionProvenanceRankingRequest,
    seed_ids: Sequence[str],
    *,
    config: ExecutionProvenanceConfig,
) -> tuple[ProvenancePathEvaluation, ...]:
    adjacency = _directed_adjacency(request)
    node_by_id = {node.node_id: node for node in request.graph.nodes}
    candidate_ids = {candidate.item_id for candidate in request.candidates}
    invalidated = invalidated_node_ids(request)
    evaluations: list[ProvenancePathEvaluation] = []
    expansion_count = 0
    for seed_id in seed_ids:
        beam = [ProvenancePath((seed_id,), ())]
        completed: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]], ProvenancePath
        ] = {}
        for _hop in range(config.max_hops):
            expanded: list[ProvenancePath] = []
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
                    expanded.append(next_path)
                    if neighbor in candidate_ids and neighbor != seed_id:
                        completed[_path_key(next_path)] = next_path
                if expansion_count >= config.max_path_expansions:
                    break
            if not expanded:
                break
            beam = sorted(
                expanded,
                key=lambda path: _beam_key(
                    path,
                    invalidated_node_ids=invalidated,
                    node_by_id=node_by_id,
                    config=config,
                ),
            )[: config.beam_width]
            if expansion_count >= config.max_path_expansions:
                break
        for path in completed.values():
            score = score_provenance_path(
                path,
                invalidated_node_ids=invalidated,
                node_by_id=node_by_id,
                config=config,
            )
            evaluations.append(
                ProvenancePathEvaluation(
                    anchor_id=seed_id,
                    partner_id=path.node_ids[-1],
                    path=path,
                    score=score,
                )
            )
        if expansion_count >= config.max_path_expansions:
            break
    return tuple(
        sorted(
            evaluations,
            key=lambda evaluation: (
                seed_ids.index(evaluation.anchor_id),
                not evaluation.score.valid,
                -evaluation.score.path_confidence,
                len(evaluation.path.edges),
                evaluation.partner_id,
                tuple(_edge_key(edge) for edge in evaluation.path.edges),
            ),
        )
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
    invalidated_node_ids: frozenset[str],
    node_by_id: Mapping[str, ExecutionProvenanceNode],
    config: ExecutionProvenanceConfig,
) -> ProvenancePathScore:
    feeds_edges = tuple(
        edge for edge in path.edges if edge.edge_type is ProvenanceEdgeType.FEEDS
    )
    returns_edges = tuple(
        edge for edge in path.edges if edge.edge_type is ProvenanceEdgeType.RETURNS
    )
    binding_valid = len(feeds_edges) == 1 and all(
        binding_matches_endpoints(edge, node_by_id) for edge in feeds_edges
    )
    completeness_valid = len(feeds_edges) == 1 and len(returns_edges) == 1
    lifecycle_valid = not any(
        node_id in invalidated_node_ids for node_id in path.node_ids
    )
    semantic_weights = [edge.weight for edge in feeds_edges]
    confidence = (
        math.exp(
            sum(math.log(max(weight, 1e-300)) for weight in semantic_weights)
            / len(semantic_weights)
        )
        if semantic_weights
        else 0.0
    )
    confidence *= math.exp(-config.hop_penalty * max(0, len(path.edges) - 2))
    rejection_reason = None
    if not completeness_valid:
        rejection_reason = "incomplete_path"
    elif not binding_valid:
        rejection_reason = "binding_mismatch"
    elif not lifecycle_valid:
        rejection_reason = "invalidated_path"
    return ProvenancePathScore(
        path_confidence=confidence,
        binding_valid=binding_valid,
        completeness_valid=completeness_valid,
        lifecycle_valid=lifecycle_valid,
        valid=(binding_valid and completeness_valid and lifecycle_valid),
        rejection_reason=rejection_reason,
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


def _beam_key(
    path: ProvenancePath,
    *,
    invalidated_node_ids: frozenset[str],
    node_by_id: Mapping[str, ExecutionProvenanceNode],
    config: ExecutionProvenanceConfig,
) -> tuple[bool, float, int, tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    score = score_provenance_path(
        path,
        invalidated_node_ids=invalidated_node_ids,
        node_by_id=node_by_id,
        config=config,
    )
    return (
        not score.lifecycle_valid,
        -score.path_confidence,
        len(path.edges),
        path.node_ids,
        tuple(_edge_key(edge) for edge in path.edges),
    )


def _path_key(
    path: ProvenancePath,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    return path.node_ids, tuple(_edge_key(edge) for edge in path.edges)


def _edge_key(edge: ExecutionProvenanceEdge) -> tuple[str, str, str]:
    return edge.source, edge.target, edge.edge_type.value


__all__ = [
    "DEPENDENCY_EDGE_TYPES",
    "ProvenancePath",
    "ProvenancePathEvaluation",
    "ProvenancePathScore",
    "invalidated_node_ids",
    "score_provenance_path",
    "search_provenance_paths",
]
