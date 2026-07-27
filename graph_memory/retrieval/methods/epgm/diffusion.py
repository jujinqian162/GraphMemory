"""Query-conditioned typed transitions and deterministic Personalized PageRank."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from graph_memory.embeddings import SentenceEncoder
from graph_memory.graphs.provenance import (
    ExecutionProvenanceGraph,
    ProvenanceEdgeType,
    binding_matches_endpoints,
)
from graph_memory.retrieval.methods.epgm.config import EpgmRetrieverConfig


@dataclass(frozen=True)
class RelationAffinity:
    edge_type: str
    similarity: float
    affinity: float


@dataclass(frozen=True)
class TypedTransition:
    source: str
    target: str
    edge_type: str
    direction: str  # forward | reverse
    recorded_weight: float
    relation_affinity: float
    probability: float
    cost: float


@dataclass(frozen=True)
class PprResult:
    node_scores: Mapping[str, float]
    teleport: Mapping[str, float]
    relation_affinities: tuple[RelationAffinity, ...]
    transitions: tuple[TypedTransition, ...]
    iterations: int
    residual: float
    converged: bool


def encode_relation_vectors(
    encoder: SentenceEncoder,
    config: EpgmRetrieverConfig,
    *,
    passage_prefix: str,
    batch_size: int,
) -> NDArray[np.float64]:
    edge_types = sorted(config.relation_descriptions)
    texts = [passage_prefix + config.relation_descriptions[edge_type] for edge_type in edge_types]
    matrix = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] != len(edge_types) or matrix.shape[1] <= 0:
        raise ValueError("relation encoder returned an invalid embedding matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("relation encoder returned non-finite embeddings")
    return matrix


def encode_query_vector(
    encoder: SentenceEncoder,
    query_text: str,
    *,
    query_prefix: str,
    batch_size: int,
) -> NDArray[np.float64]:
    matrix = np.asarray(
        encoder.encode(
            [query_prefix + query_text],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] != 1 or matrix.shape[1] <= 0:
        raise ValueError("query encoder returned an invalid embedding matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("query encoder returned non-finite embeddings")
    return matrix[0]


def query_relation_affinities(
    query_vector: NDArray[np.float64],
    relation_vectors: NDArray[np.float64],
    present_edge_types: frozenset[str],
    config: EpgmRetrieverConfig,
) -> tuple[RelationAffinity, ...]:
    """Softmax query/relation cosine over relation types present in the graph."""

    edge_types = sorted(config.relation_descriptions)
    if relation_vectors.shape != (len(edge_types), query_vector.shape[0]):
        raise ValueError("query and relation embedding dimensions do not match")
    present = [edge_type for edge_type in edge_types if edge_type in present_edge_types]
    if not present:
        return ()
    index_by_type = {edge_type: index for index, edge_type in enumerate(edge_types)}
    similarities = np.asarray(
        [float(relation_vectors[index_by_type[edge_type]] @ query_vector) for edge_type in present],
        dtype=float,
    )
    logits = similarities / config.relation_temperature
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    weights /= float(np.sum(weights))
    return tuple(
        RelationAffinity(edge_type, float(similarity), float(affinity))
        for edge_type, similarity, affinity in zip(
            present, similarities, weights, strict=True
        )
    )


def dense_teleport(
    dense_scores: Mapping[str, float],
    graph: ExecutionProvenanceGraph,
) -> dict[str, float]:
    """Strictly positive Dense relevance over graph-backed candidates.

    Raw normalized-encoder cosine scores occupy a narrow interval, so a
    unit-temperature softmax is almost uniform and erases the semantic ranking
    before diffusion. Min-max relevance preserves the observed query-specific
    dynamic range; a fixed numerical floor keeps every graph-backed request
    candidate reachable without introducing a dataset-tuned temperature.
    """

    node_ids = {node.node_id for node in graph.nodes}
    candidates = sorted(node_id for node_id in dense_scores if node_id in node_ids)
    if not candidates:
        raise ValueError("no request candidate exists in the provenance graph")
    values = np.asarray([float(dense_scores[node_id]) for node_id in candidates])
    if not np.isfinite(values).all():
        raise ValueError("dense scores must be finite")
    low = float(np.min(values))
    span = float(np.max(values)) - low
    if span <= 0.0:
        weights = np.ones_like(values)
    else:
        weights = (values - low) / span
        weights += 1e-6
    weights /= float(np.sum(weights))
    return {
        node_id: float(weight)
        for node_id, weight in zip(candidates, weights, strict=True)
    }


def build_typed_transitions(
    graph: ExecutionProvenanceGraph,
    relation_affinities: Sequence[RelationAffinity],
    config: EpgmRetrieverConfig,
) -> tuple[TypedTransition, ...]:
    """Build row-stochastic typed arcs with source-local recorded confidence."""

    affinity_by_type = {item.edge_type: item.affinity for item in relation_affinities}
    node_by_id = {node.node_id: node for node in graph.nodes}
    undirected_degree: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        undirected_degree[edge.source] += 1
        undirected_degree[edge.target] += 1

    raw_by_source: dict[
        str, list[tuple[str, str, str, str, float, float, float]]
    ] = defaultdict(list)
    for edge in sorted(
        graph.edges,
        key=lambda item: (item.source, item.target, item.edge_type.value),
    ):
        # A recorded zero is explicit abstention, not a topology-only link; it
        # disables both stored and reverse traversal.
        if float(edge.weight) <= 0.0:
            continue
        if edge.edge_type is ProvenanceEdgeType.FEEDS and not binding_matches_endpoints(
            edge, node_by_id
        ):
            continue
        relation = affinity_by_type.get(edge.edge_type.value)
        if relation is None or relation <= 0.0:
            continue
        prior = max(config.edge_prior(edge.edge_type.value), config.min_edge_prior)
        forward_target_penalty = (
            1.0 + undirected_degree[edge.target]
        ) ** (-config.hub_degree_exponent)
        forward_strength = prior * relation * float(edge.weight) * forward_target_penalty
        if forward_strength > 0.0:
            raw_by_source[edge.source].append(
                (
                    edge.target,
                    edge.edge_type.value,
                    "forward",
                    edge.source,
                    float(edge.weight),
                    relation,
                    forward_strength,
                )
            )
        reverse_target_penalty = (
            1.0 + undirected_degree[edge.source]
        ) ** (-config.hub_degree_exponent)
        # Recorded confidence is generated in the stored source direction and
        # is therefore an identity factor for a reverse traversal.
        reverse_strength = (
            prior
            * relation
            * config.reverse_edge_factor
            * reverse_target_penalty
        )
        if reverse_strength > 0.0:
            raw_by_source[edge.target].append(
                (
                    edge.source,
                    edge.edge_type.value,
                    "reverse",
                    edge.source,
                    float(edge.weight),
                    relation,
                    reverse_strength,
                )
            )

    transitions: list[TypedTransition] = []
    for source in sorted(raw_by_source):
        arcs = sorted(
            raw_by_source[source], key=lambda item: (item[0], item[1], item[2], item[3])
        )
        total = math.fsum(item[6] for item in arcs)
        if not math.isfinite(total) or total <= 0.0:
            continue
        for target, edge_type, direction, _stored_source, weight, relation, strength in arcs:
            probability = strength / total
            transitions.append(
                TypedTransition(
                    source=source,
                    target=target,
                    edge_type=edge_type,
                    direction=direction,
                    recorded_weight=weight,
                    relation_affinity=relation,
                    probability=probability,
                    cost=(
                        -math.log(max(probability, 1e-300))
                        + config.connector_hop_cost
                    ),
                )
            )
    return tuple(transitions)


def personalized_pagerank(
    graph: ExecutionProvenanceGraph,
    teleport: Mapping[str, float],
    relation_affinities: Sequence[RelationAffinity],
    config: EpgmRetrieverConfig,
) -> PprResult:
    nodes = tuple(sorted(node.node_id for node in graph.nodes))
    teleport_full = {node_id: float(teleport.get(node_id, 0.0)) for node_id in nodes}
    total_teleport = math.fsum(teleport_full.values())
    if not math.isclose(total_teleport, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("teleport distribution must sum to one")
    transitions = build_typed_transitions(graph, relation_affinities, config)
    outgoing: dict[str, list[TypedTransition]] = defaultdict(list)
    for transition in transitions:
        outgoing[transition.source].append(transition)
    for source, row in outgoing.items():
        row_total = math.fsum(item.probability for item in row)
        if not math.isclose(row_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"transition row does not normalize: source={source}")

    scores = dict(teleport_full)
    residual = math.inf
    converged = False
    iterations = 0
    alpha = config.ppr_alpha
    for iterations in range(1, config.ppr_max_iterations + 1):
        dangling = math.fsum(scores[node_id] for node_id in nodes if node_id not in outgoing)
        next_scores = {
            node_id: (1.0 - alpha) * teleport_full[node_id]
            + alpha * dangling * teleport_full[node_id]
            for node_id in nodes
        }
        for source in nodes:
            source_mass = scores[source]
            for transition in outgoing.get(source, ()):
                next_scores[transition.target] += (
                    alpha * source_mass * transition.probability
                )
        total = math.fsum(next_scores.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("PPR produced invalid probability mass")
        next_scores = {node_id: value / total for node_id, value in next_scores.items()}
        residual = math.fsum(abs(next_scores[node_id] - scores[node_id]) for node_id in nodes)
        scores = next_scores
        if residual <= config.ppr_tolerance:
            converged = True
            break
    return PprResult(
        node_scores=scores,
        teleport=teleport_full,
        relation_affinities=tuple(relation_affinities),
        transitions=transitions,
        iterations=iterations,
        residual=residual,
        converged=converged,
    )


__all__ = [
    "PprResult",
    "RelationAffinity",
    "TypedTransition",
    "build_typed_transitions",
    "dense_teleport",
    "encode_query_vector",
    "encode_relation_vectors",
    "personalized_pagerank",
    "query_relation_affinities",
]
