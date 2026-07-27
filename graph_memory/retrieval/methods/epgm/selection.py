"""Deterministic budgeted Steiner-style evidence subgraph extraction."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.retrieval.methods.epgm.config import EpgmRetrieverConfig
from graph_memory.retrieval.methods.epgm.diffusion import PprResult, TypedTransition


_EMITTABLE_RELATIONS = frozenset(
    {
        "feeds",
        "depends_on",
        "grounds",
        "supports",
        "verifies",
        "contradicts",
        "invalidates",
        "supersedes",
        "affects",
    }
)


@dataclass(frozen=True)
class CandidatePrize:
    node_id: str
    dense_component: float
    ppr_component: float
    prize: float


@dataclass(frozen=True)
class SelectionStep:
    anchor_id: str
    target_id: str
    path_node_ids: tuple[str, ...]
    transitions: tuple[TypedTransition, ...]
    added_candidate_ids: tuple[str, ...]
    displaced_candidate_ids: tuple[str, ...]
    prize_gain: float
    edge_cost: float
    displacement_cost: float
    marginal_gain: float


@dataclass(frozen=True)
class SelectedCandidateEdge:
    source: str
    target: str
    edge_type: str
    confidence: float


@dataclass(frozen=True)
class SelectedSubgraph:
    candidate_ids: tuple[str, ...]
    connector_ids: tuple[str, ...]
    transitions: tuple[TypedTransition, ...]
    steps: tuple[SelectionStep, ...]
    candidate_edges: tuple[SelectedCandidateEdge, ...]
    objective: float
    exact_dense_fallback: bool


def candidate_prizes(
    dense_scores: Mapping[str, float],
    ppr: PprResult,
    config: EpgmRetrieverConfig,
) -> tuple[CandidatePrize, ...]:
    ids = sorted(dense_scores)
    dense_values = [float(dense_scores[node_id]) for node_id in ids]
    dense_low, dense_high = min(dense_values), max(dense_values)
    ppr_values = [float(ppr.node_scores.get(node_id, 0.0)) for node_id in ids]
    ppr_high = max(ppr_values) if ppr_values else 0.0
    prizes: list[CandidatePrize] = []
    for node_id, dense_value, ppr_value in zip(
        ids, dense_values, ppr_values, strict=True
    ):
        dense_component = (
            1.0
            if dense_high <= dense_low
            else (dense_value - dense_low) / (dense_high - dense_low)
        )
        ppr_component = 0.0 if ppr_high <= 0.0 else ppr_value / ppr_high
        prize = max(
            0.0,
            config.dense_prize_weight * dense_component
            + config.ppr_prize_weight * ppr_component
            - config.candidate_inclusion_cost,
        )
        prizes.append(
            CandidatePrize(node_id, dense_component, ppr_component, prize)
        )
    return tuple(prizes)


def _transition_key(item: TypedTransition) -> tuple[str, str, str, str]:
    return item.source, item.target, item.edge_type, item.direction


def _shortest_connector_paths(
    anchors: Sequence[str],
    transitions: Sequence[TypedTransition],
    *,
    zero_cost_transition_keys: frozenset[tuple[str, str, str, str]] = frozenset(),
) -> dict[str, tuple[tuple[str, ...], tuple[TypedTransition, ...]]]:
    """Shortest residual connector paths from the selected evidence anchors.

    Transitions already present in the current selected tree have zero
    *marginal* cost. Starting from every selected evidence candidate and
    traversing those zero-cost arcs exposes the complete reachable tree
    frontier while preserving a candidate anchor and a full auditable path for
    candidate-edge collapse.
    """
    outgoing: dict[str, list[TypedTransition]] = defaultdict(list)
    for transition in transitions:
        outgoing[transition.source].append(transition)
    for row in outgoing.values():
        row.sort(key=_transition_key)

    heap: list[
        tuple[
            float,
            tuple[str, ...],
            tuple[tuple[str, str, str, str], ...],
            str,
            tuple[TypedTransition, ...],
        ]
    ] = []
    best: dict[
        str,
        tuple[float, tuple[str, ...], tuple[tuple[str, str, str, str], ...]],
    ] = {}
    for anchor in sorted(anchors):
        path = (anchor,)
        heapq.heappush(heap, (0.0, path, (), anchor, ()))
        best[anchor] = (0.0, path, ())
    resolved: dict[str, tuple[tuple[str, ...], tuple[TypedTransition, ...]]] = {}
    while heap:
        cost, node_path, edge_keys, node_id, edge_path = heapq.heappop(heap)
        if best.get(node_id) != (cost, node_path, edge_keys):
            continue
        resolved[node_id] = (node_path, edge_path)
        for transition in outgoing.get(node_id, ()):
            if transition.target in node_path:
                continue
            transition_cost = (
                0.0
                if _transition_key(transition) in zero_cost_transition_keys
                else transition.cost
            )
            next_cost = cost + transition_cost
            next_nodes = (*node_path, transition.target)
            next_edge_keys = (*edge_keys, _transition_key(transition))
            current = best.get(transition.target)
            if current is not None and current <= (
                next_cost,
                next_nodes,
                next_edge_keys,
            ):
                continue
            best[transition.target] = (next_cost, next_nodes, next_edge_keys)
            heapq.heappush(
                heap,
                (
                    next_cost,
                    next_nodes,
                    next_edge_keys,
                    transition.target,
                    (*edge_path, transition),
                ),
            )
    return resolved


def _candidate_edges(
    steps: Sequence[SelectionStep],
    candidate_ids: frozenset[str],
) -> tuple[SelectedCandidateEdge, ...]:
    edges: list[SelectedCandidateEdge] = []
    seen: set[tuple[str, str]] = set()
    for step in steps:
        positions = [
            index
            for index, node_id in enumerate(step.path_node_ids)
            if node_id in candidate_ids
        ]
        for left, right in zip(positions, positions[1:], strict=False):
            source = step.path_node_ids[left]
            target = step.path_node_ids[right]
            segment = step.transitions[left:right]
            if not segment or not any(
                item.edge_type in _EMITTABLE_RELATIONS for item in segment
            ):
                # contains/invokes/returns/precedes can connect and rank
                # structural evidence, but are not contracted output
                # dependencies and must not inflate shared edge metrics.
                continue
            directions = {item.direction for item in segment}
            if len(directions) != 1:
                # A <- X -> B is connected but does not establish A -> B or
                # B -> A. Keep it in the native subgraph and emit no fabricated
                # candidate dependency.
                continue
            if directions == {"reverse"}:
                source, target = target, source
            key = (source, target)
            if source == target or key in seen:
                continue
            seen.add(key)
            semantic_type = (
                segment[0].edge_type
                if len(segment) == 1
                else (
                    "feeds"
                    if any(item.edge_type == "feeds" for item in segment)
                    else "depends_on"
                )
            )
            edges.append(
                SelectedCandidateEdge(
                    source=source,
                    target=target,
                    edge_type=semantic_type,
                    confidence=math.exp(-math.fsum(item.cost for item in segment)),
                )
            )
    return tuple(edges)


def select_budgeted_subgraph(
    dense_scores: Mapping[str, float],
    ppr: PprResult,
    *,
    top_k: int,
    config: EpgmRetrieverConfig,
) -> tuple[SelectedSubgraph, tuple[CandidatePrize, ...]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    prizes = candidate_prizes(dense_scores, ppr, config)
    prize_by_id = {item.node_id: item for item in prizes}
    dense_order = sorted(dense_scores, key=lambda item: (-dense_scores[item], item))
    dense_rank = {
        node_id: index for index, node_id in enumerate(dense_order, start=1)
    }
    candidate_set = frozenset(dense_scores)
    evidence_budget = min(top_k, len(candidate_set))
    incumbent_set = set(dense_order[:evidence_budget])
    # The connected challenger starts from the strongest prize already present
    # in the Dense top-k incumbent. Starting from an out-of-budget graph-central
    # node would promote it without paying the evidence it displaces.
    root = min(
        (item for item in prizes if item.node_id in incumbent_set),
        key=lambda item: (-item.prize, dense_rank[item.node_id], item.node_id),
    ).node_id
    selected_candidates: list[str] = [root]
    selected_set = {root}
    selected_transitions: dict[tuple[str, str, str, str], TypedTransition] = {}
    selected_nodes = {root}
    steps: list[SelectionStep] = []
    objective = prize_by_id[root].prize

    while len(selected_set) < evidence_budget:
        proposals: list[SelectionStep] = []
        shortest = _shortest_connector_paths(
            selected_candidates,
            ppr.transitions,
            zero_cost_transition_keys=frozenset(selected_transitions),
        )
        for target in sorted(candidate_set - selected_set):
            found = shortest.get(target)
            if found is None:
                continue
            node_path, edge_path = found
            added = tuple(
                node_id
                for node_id in node_path
                if node_id in candidate_set and node_id not in selected_set
            )
            if not added or len(selected_set) + len(added) > evidence_budget:
                continue
            outside_incumbent = tuple(
                node_id for node_id in added if node_id not in incumbent_set
            )
            displacement_pool = sorted(
                incumbent_set - selected_set - set(added),
                key=lambda node_id: (
                    prize_by_id[node_id].prize,
                    -dense_rank[node_id],
                    node_id,
                ),
            )
            if len(displacement_pool) < len(outside_incumbent):
                continue
            displaced = tuple(displacement_pool[: len(outside_incumbent)])
            prize_gain = math.fsum(prize_by_id[node_id].prize for node_id in added)
            new_edges = [
                edge
                for edge in edge_path
                if _transition_key(edge) not in selected_transitions
            ]
            edge_cost = math.fsum(edge.cost for edge in new_edges)
            displacement_cost = math.fsum(
                prize_by_id[node_id].prize for node_id in displaced
            )
            replacement_gain = (
                math.fsum(
                    prize_by_id[node_id].prize for node_id in outside_incumbent
                )
                - displacement_cost
            )
            # Candidates already present in Dense top-k may justify connecting
            # their own subgraph, but their prize must not subsidize a weaker
            # out-of-budget candidate. Every membership-changing bundle needs
            # positive replacement utility on its own.
            if outside_incumbent and replacement_gain <= 0.0:
                continue
            marginal = (
                prize_gain
                - config.selection_edge_cost_weight * edge_cost
                - displacement_cost
            )
            proposals.append(
                SelectionStep(
                    anchor_id=node_path[0],
                    target_id=target,
                    path_node_ids=node_path,
                    transitions=edge_path,
                    added_candidate_ids=added,
                    displaced_candidate_ids=displaced,
                    prize_gain=prize_gain,
                    edge_cost=edge_cost,
                    displacement_cost=displacement_cost,
                    marginal_gain=marginal,
                )
            )
        eligible = [
            proposal
            for proposal in proposals
            if proposal.marginal_gain > config.selection_min_gain
        ]
        if not eligible:
            break
        winner = min(
            eligible,
            key=lambda item: (
                -item.marginal_gain,
                item.edge_cost,
                dense_rank[item.target_id],
                item.path_node_ids,
                tuple(_transition_key(edge) for edge in item.transitions),
            ),
        )
        steps.append(winner)
        objective += winner.marginal_gain
        for node_id in winner.path_node_ids:
            selected_nodes.add(node_id)
        for node_id in winner.displaced_candidate_ids:
            incumbent_set.remove(node_id)
        for node_id in winner.added_candidate_ids:
            selected_set.add(node_id)
            selected_candidates.append(node_id)
            incumbent_set.add(node_id)
        for edge in winner.transitions:
            selected_transitions[_transition_key(edge)] = edge

    candidate_edges = _candidate_edges(steps, candidate_set)
    fallback = len(selected_set) < 2 or not steps
    if fallback:
        return (
            SelectedSubgraph((), (), (), (), (), 0.0, True),
            prizes,
        )
    connectors = tuple(sorted(selected_nodes - candidate_set))
    return (
        SelectedSubgraph(
            candidate_ids=tuple(selected_candidates),
            connector_ids=connectors,
            transitions=tuple(
                selected_transitions[key] for key in sorted(selected_transitions)
            ),
            steps=tuple(steps),
            candidate_edges=candidate_edges,
            objective=objective,
            exact_dense_fallback=False,
        ),
        prizes,
    )


__all__ = [
    "CandidatePrize",
    "SelectedCandidateEdge",
    "SelectedSubgraph",
    "SelectionStep",
    "candidate_prizes",
    "select_budgeted_subgraph",
]
