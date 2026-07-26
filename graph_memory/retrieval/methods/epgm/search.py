"""Single bounded typed path search shared by both EPGM presets.

The traversal, edge scope, path scoring, and gating axes come from
``EpgmRetrieverConfig``; the search itself is one implementation. The
``dependency_path`` preset is the same code with a directed walk, the
dependency edge subset, ``feeds``-geometric scoring, and hard schema gating.

Recorded edge weights enter through :func:`effective_edge_weights`, which
normalises them per edge type and returns nothing for any edge type whose
weights have no variance. Multiplying by the resulting ``w_eff`` is therefore
an identity operation on graphs that store a constant placeholder weight, so
weight sensitivity needs no dataset-specific configuration.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
    ProvenanceEdgeType,
    binding_matches_endpoints,
)
from graph_memory.retrieval.methods.epgm.config import (
    DEPENDENCY_EDGE_TYPES,
    EpgmRetrieverConfig,
    HUB_NODE_TYPES,
    INVALID_LIFECYCLE_STATES,
    NON_TRAVERSABLE_EDGE_TYPES,
    REVISION_EDGE_TYPES,
    WEIGHT_INFORMATIVE_VARIANCE,
)


@dataclass(frozen=True)
class EpgmPathStep:
    source: str
    target: str
    edge_type: str
    direction: str  # "forward" | "reverse"
    weight: float


@dataclass(frozen=True)
class EpgmGateReport:
    """Schema-gate outcome. Always populated, even when gating is disabled."""

    binding_valid: bool
    completeness_valid: bool
    lifecycle_valid: bool
    valid: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class EpgmPath:
    seed_id: str
    target_id: str
    node_ids: tuple[str, ...]
    steps: tuple[EpgmPathStep, ...]
    score: float
    gate: EpgmGateReport


@dataclass(frozen=True)
class _Arc:
    neighbor: str
    edge: ExecutionProvenanceEdge
    direction: str  # "forward" | "reverse"


_PASS = EpgmGateReport(
    binding_valid=True,
    completeness_valid=True,
    lifecycle_valid=True,
    valid=True,
    rejection_reason=None,
)


def invalidated_node_ids(graph: ExecutionProvenanceGraph) -> frozenset[str]:
    invalidated = {
        edge.target for edge in graph.edges if edge.edge_type in REVISION_EDGE_TYPES
    }
    for node in graph.nodes:
        lifecycle_state = node.metadata.get("lifecycle_state")
        if (
            isinstance(lifecycle_state, str)
            and lifecycle_state.casefold() in INVALID_LIFECYCLE_STATES
        ) or node.metadata.get("valid") is False:
            invalidated.add(node.node_id)
    return frozenset(invalidated)


def _adjacency(
    graph: ExecutionProvenanceGraph,
    *,
    node_type_by_id: Mapping[str, str],
    config: EpgmRetrieverConfig,
) -> dict[str, tuple[_Arc, ...]]:
    """Typed adjacency honouring the traversal and edge-scope axes.

    ``contains`` edges are always dropped and hub nodes (task/agent) are never
    expanded *from*, so they can only ever be a terminal node on a path. A
    directed walk emits forward arcs only; a bidirectional walk also emits a
    reverse arc that remembers the stored orientation.
    """

    dependency_only = config.edge_scope == "dependency"
    bidirectional = config.traversal == "bidirectional"
    mutable: dict[str, list[_Arc]] = defaultdict(list)
    for edge in graph.edges:
        if edge.edge_type.value in NON_TRAVERSABLE_EDGE_TYPES:
            continue
        if dependency_only and edge.edge_type not in DEPENDENCY_EDGE_TYPES:
            continue
        if node_type_by_id.get(edge.source) not in HUB_NODE_TYPES:
            mutable[edge.source].append(_Arc(edge.target, edge, "forward"))
        if bidirectional and node_type_by_id.get(edge.target) not in HUB_NODE_TYPES:
            mutable[edge.target].append(_Arc(edge.source, edge, "reverse"))
    return {
        node_id: tuple(
            sorted(arcs, key=lambda arc: (arc.neighbor, arc.edge.edge_type.value))
        )
        for node_id, arcs in mutable.items()
    }


def effective_edge_weights(
    graph: ExecutionProvenanceGraph,
    config: EpgmRetrieverConfig,
) -> Mapping[tuple[str, str, str], float]:
    """Per-edge ``w_eff``, normalised within each edge type.

    A recorded weight only carries information relative to the other weights of
    the *same* edge type: ``feeds`` weights produced by a semantic scorer are
    comparable to each other, but not to the constant ``1.0`` that structural
    ``contains``/``invokes`` edges carry. So each edge type is normalised
    independently and any type whose weights have (near) zero variance yields
    exactly ``1.0``, making this factor an identity operation there.

    The consequence is that no dataset switch is needed: a scorer-calibrated
    graph gets full weight sensitivity, and a recorded agent trace that fills a
    constant placeholder weight everywhere falls back to pure type priors
    because every edge type collapses to ``1.0``.
    """

    if not config.weight_aware:
        return {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for edge in graph.edges:
        grouped[edge.edge_type.value].append(float(edge.weight))

    informative: dict[str, tuple[float, float]] = {}
    for edge_type, weights in grouped.items():
        if len(weights) < 2:
            continue
        mean = sum(weights) / len(weights)
        variance = sum((weight - mean) ** 2 for weight in weights) / len(weights)
        if variance <= WEIGHT_INFORMATIVE_VARIANCE:
            continue
        low, high = min(weights), max(weights)
        if high - low <= 0.0:
            continue
        informative[edge_type] = (low, high)

    if not informative:
        return {}
    floor = config.min_effective_weight
    span = 1.0 - floor
    effective: dict[tuple[str, str, str], float] = {}
    for edge in graph.edges:
        bounds = informative.get(edge.edge_type.value)
        if bounds is None:
            continue
        low, high = bounds
        scaled = (float(edge.weight) - low) / (high - low)
        effective[(edge.source, edge.target, edge.edge_type.value)] = (
            floor + span * scaled
        )
    return effective


def _arc_weight(
    arc: _Arc,
    config: EpgmRetrieverConfig,
    effective_weights: Mapping[tuple[str, str, str], float],
) -> float:
    prior = config.edge_prior(arc.edge.edge_type.value)
    # Absent from the map means "carries no information here", i.e. w_eff = 1.
    prior *= effective_weights.get(
        (arc.edge.source, arc.edge.target, arc.edge.edge_type.value), 1.0
    )
    if arc.direction == "reverse":
        prior *= config.reverse_edge_factor
    return prior


def _type_prior_score(
    *,
    running: float,
    base: float,
    arc: _Arc,
    hop: int,
    config: EpgmRetrieverConfig,
    effective_weights: Mapping[tuple[str, str, str], float],
) -> float:
    step = _arc_weight(arc, config, effective_weights) * (config.hop_decay**hop)
    return running * step if running > 0.0 else step * base


def _feeds_geometric_score(
    steps: Sequence[EpgmPathStep],
    config: EpgmRetrieverConfig,
) -> float:
    weights = [
        step.weight
        for step in steps
        if step.edge_type == ProvenanceEdgeType.FEEDS.value
    ]
    if not weights:
        return 0.0
    confidence = math.exp(
        sum(math.log(max(weight, 1e-300)) for weight in weights) / len(weights)
    )
    return confidence * math.exp(-config.hop_penalty * max(0, len(steps) - 2))


def _gate(
    steps: Sequence[EpgmPathStep],
    node_ids: Sequence[str],
    *,
    edge_by_key: Mapping[tuple[str, str, str], ExecutionProvenanceEdge],
    node_by_id: Mapping[str, ExecutionProvenanceNode],
    invalidated: frozenset[str],
) -> EpgmGateReport:
    feeds = [
        step for step in steps if step.edge_type == ProvenanceEdgeType.FEEDS.value
    ]
    returns = [
        step for step in steps if step.edge_type == ProvenanceEdgeType.RETURNS.value
    ]
    binding_valid = len(feeds) == 1 and all(
        binding_matches_endpoints(
            edge_by_key[(step.source, step.target, step.edge_type)], node_by_id
        )
        for step in feeds
    )
    completeness_valid = len(feeds) == 1 and len(returns) == 1
    lifecycle_valid = not any(node_id in invalidated for node_id in node_ids)
    rejection_reason = None
    if not completeness_valid:
        rejection_reason = "incomplete_path"
    elif not binding_valid:
        rejection_reason = "binding_mismatch"
    elif not lifecycle_valid:
        rejection_reason = "invalidated_path"
    return EpgmGateReport(
        binding_valid=binding_valid,
        completeness_valid=completeness_valid,
        lifecycle_valid=lifecycle_valid,
        valid=binding_valid and completeness_valid and lifecycle_valid,
        rejection_reason=rejection_reason,
    )


def search_epgm_paths(
    graph: ExecutionProvenanceGraph,
    seed_ids: Sequence[str],
    seed_relevance: Mapping[str, float],
    *,
    candidate_ids: frozenset[str],
    config: EpgmRetrieverConfig,
) -> tuple[EpgmPath, ...]:
    """Bounded typed beam search from each dense seed.

    Returns, per seed, either the best-scoring path reaching each candidate
    node (gating disabled) or every distinct completed path (schema gating,
    where each rejected path is audit evidence for the gate).
    Invalidated/superseded nodes stay reachable in every configuration; when
    schema gating is on they are reported as ``invalidated_path`` instead of
    being pruned, so revision-audit queries can still surface them.
    """

    node_type_by_id = {node.node_id: node.node_type.value for node in graph.nodes}
    node_by_id = {node.node_id: node for node in graph.nodes}
    edge_by_key = {
        (edge.source, edge.target, edge.edge_type.value): edge for edge in graph.edges
    }
    invalidated = invalidated_node_ids(graph)
    adjacency = _adjacency(graph, node_type_by_id=node_type_by_id, config=config)
    effective_weights = effective_edge_weights(graph, config)
    gating = config.gating == "schema"
    feeds_scored = config.path_score == "feeds_geometric"

    results: list[EpgmPath] = []
    expansion_count = 0

    for seed_id in seed_ids:
        base = float(seed_relevance.get(seed_id, 0.0))
        beam: list[tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float]] = [
            ((seed_id,), (), base)
        ]
        # gating on: keep every distinct completed path so each rejection is
        # auditable. gating off: keep only the best path per candidate.
        best: dict[
            object, tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float]
        ] = {}
        for hop in range(config.max_hops):
            expanded: list[
                tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float]
            ] = []
            for node_ids, steps, score in beam:
                tail = node_ids[-1]
                for arc in adjacency.get(tail, ()):  # noqa: PLR1702
                    if expansion_count >= config.max_path_expansions:
                        break
                    if arc.neighbor in node_ids:
                        continue
                    expansion_count += 1
                    next_ids = (*node_ids, arc.neighbor)
                    next_steps = (
                        *steps,
                        EpgmPathStep(
                            source=arc.edge.source,
                            target=arc.edge.target,
                            edge_type=arc.edge.edge_type.value,
                            direction=arc.direction,
                            weight=float(arc.edge.weight),
                        ),
                    )
                    next_score = (
                        _feeds_geometric_score(next_steps, config)
                        if feeds_scored
                        else _type_prior_score(
                            running=score,
                            base=base,
                            arc=arc,
                            hop=hop,
                            config=config,
                            effective_weights=effective_weights,
                        )
                    )
                    entry = (next_ids, next_steps, next_score)
                    expanded.append(entry)
                    if arc.neighbor in candidate_ids and arc.neighbor != seed_id:
                        key: object = next_ids if gating else arc.neighbor
                        current = best.get(key)
                        if current is None or next_score > current[2]:
                            best[key] = entry
                if expansion_count >= config.max_path_expansions:
                    break
            if not expanded:
                break
            beam = sorted(expanded, key=_beam_key(invalidated, gating))[
                : config.beam_width
            ]
            if expansion_count >= config.max_path_expansions:
                break

        for node_ids, steps, score in best.values():
            gate = (
                _gate(
                    steps,
                    node_ids,
                    edge_by_key=edge_by_key,
                    node_by_id=node_by_id,
                    invalidated=invalidated,
                )
                if gating
                else _PASS
            )
            results.append(
                EpgmPath(
                    seed_id=seed_id,
                    target_id=node_ids[-1],
                    node_ids=node_ids,
                    steps=steps,
                    score=score,
                    gate=gate,
                )
            )

    return tuple(
        sorted(
            results,
            key=lambda path: (
                not path.gate.valid,
                -path.score,
                len(path.steps),
                path.seed_id,
                path.target_id,
            ),
        )
    )


def _beam_key(invalidated: frozenset[str], gating: bool):
    def key(
        entry: tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float],
    ) -> tuple[bool, float, int, tuple[str, ...]]:
        node_ids, steps, score = entry
        lifecycle_penalty = gating and any(
            node_id in invalidated for node_id in node_ids
        )
        return (lifecycle_penalty, -score, len(steps), node_ids)

    return key


__all__ = [
    "EpgmGateReport",
    "EpgmPath",
    "EpgmPathStep",
    "effective_edge_weights",
    "invalidated_node_ids",
    "search_epgm_paths",
]
