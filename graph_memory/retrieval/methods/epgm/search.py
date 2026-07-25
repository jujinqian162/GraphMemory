from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceGraph,
    ExecutionProvenanceNode,
)
from graph_memory.retrieval.methods.epgm.config import (
    EpgmRetrieverConfig,
    HUB_NODE_TYPES,
    NON_TRAVERSABLE_EDGE_TYPES,
)


@dataclass(frozen=True)
class EpgmPathStep:
    source: str
    target: str
    edge_type: str
    direction: str  # "forward" | "reverse"
    weight: float


@dataclass(frozen=True)
class EpgmPath:
    seed_id: str
    target_id: str
    node_ids: tuple[str, ...]
    steps: tuple[EpgmPathStep, ...]
    score: float


@dataclass(frozen=True)
class _Arc:
    neighbor: str
    edge: ExecutionProvenanceEdge
    direction: str  # "forward" | "reverse"


def _bidirectional_adjacency(
    graph: ExecutionProvenanceGraph,
    *,
    node_type_by_id: Mapping[str, str],
    config: EpgmRetrieverConfig,
) -> dict[str, tuple[_Arc, ...]]:
    """Undirected walk over typed edges, remembering the original direction.

    `contains` edges are dropped entirely and hub nodes (task/agent) are never
    expanded *from*, so they can only ever be a terminal node on a path.
    """

    mutable: dict[str, list[_Arc]] = defaultdict(list)
    for edge in graph.edges:
        if edge.edge_type.value in NON_TRAVERSABLE_EDGE_TYPES:
            continue
        # forward: source -> target
        if node_type_by_id.get(edge.source) not in HUB_NODE_TYPES:
            mutable[edge.source].append(_Arc(edge.target, edge, "forward"))
        # reverse: target -> source
        if node_type_by_id.get(edge.target) not in HUB_NODE_TYPES:
            mutable[edge.target].append(_Arc(edge.source, edge, "reverse"))
    return {
        node_id: tuple(
            sorted(arcs, key=lambda arc: (arc.neighbor, arc.edge.edge_type.value))
        )
        for node_id, arcs in mutable.items()
    }


def _arc_weight(arc: _Arc, config: EpgmRetrieverConfig) -> float:
    prior = config.edge_prior(arc.edge.edge_type.value)
    if arc.direction == "reverse":
        prior *= config.reverse_edge_factor
    return prior


def search_epgm_paths(
    graph: ExecutionProvenanceGraph,
    seed_ids: Sequence[str],
    seed_relevance: Mapping[str, float],
    *,
    candidate_ids: frozenset[str],
    config: EpgmRetrieverConfig,
) -> tuple[EpgmPath, ...]:
    """Bounded typed beam search from each dense seed.

    Returns the best-scoring path reaching every candidate node, per seed. The
    path score multiplies the seed's dense relevance by a decaying product of
    typed edge priors. invalidated/superseded nodes are NOT excluded; they are
    reachable like any other node (lifecycle is reported, not filtered).
    """

    node_type_by_id = {node.node_id: node.node_type.value for node in graph.nodes}
    adjacency = _bidirectional_adjacency(
        graph, node_type_by_id=node_type_by_id, config=config
    )
    results: list[EpgmPath] = []
    expansion_count = 0

    for seed_id in seed_ids:
        base = float(seed_relevance.get(seed_id, 0.0))
        # beam entries: (node_ids, steps, running_score)
        beam: list[tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float]] = [
            ((seed_id,), (), base)
        ]
        best_for_target: dict[str, tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float]] = {}
        for hop in range(config.max_hops):
            expanded: list[
                tuple[tuple[str, ...], tuple[EpgmPathStep, ...], float]
            ] = []
            decay = config.hop_decay ** hop
            for node_ids, steps, score in beam:
                tail = node_ids[-1]
                for arc in adjacency.get(tail, ()):  # noqa: PLR1702
                    if expansion_count >= config.max_path_expansions:
                        break
                    if arc.neighbor in node_ids:
                        continue
                    expansion_count += 1
                    step_score = _arc_weight(arc, config) * decay
                    next_score = score * step_score if score > 0.0 else step_score * base
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
                    entry = (next_ids, next_steps, next_score)
                    expanded.append(entry)
                    if arc.neighbor in candidate_ids and arc.neighbor != seed_id:
                        current = best_for_target.get(arc.neighbor)
                        if current is None or next_score > current[2]:
                            best_for_target[arc.neighbor] = entry
                if expansion_count >= config.max_path_expansions:
                    break
            if not expanded:
                break
            beam = sorted(expanded, key=lambda item: -item[2])[: config.beam_width]
            if expansion_count >= config.max_path_expansions:
                break

        for target_id, (node_ids, steps, score) in best_for_target.items():
            results.append(
                EpgmPath(
                    seed_id=seed_id,
                    target_id=target_id,
                    node_ids=node_ids,
                    steps=steps,
                    score=score,
                )
            )

    return tuple(
        sorted(results, key=lambda path: (-path.score, path.seed_id, path.target_id))
    )


__all__ = [
    "EpgmPath",
    "EpgmPathStep",
    "search_epgm_paths",
]
