"""Bounded typed walk, partner proposals, and stable promotion.

This is the whole EPGM operator. From each leading Dense anchor it enumerates
every request candidate reachable within ``max_hops`` undirected typed steps,
scores each reachable partner, accepts at most one partner per anchor under a
single confidence threshold, and promotes accepted partners to just after their
anchor.

Enumeration is exhaustive within the hop bound, so no discovery order can starve
a reachable partner. That is the structural difference from a budgeted greedy
selector, where one long path can consume the whole selection budget before a
distant but well-connected candidate is ever proposed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.graphs.provenance import ExecutionProvenanceGraph
from graph_memory.retrieval.methods.epgm.config import (
    EMITTABLE_EDGE_TYPES,
    EpgmRetrieverConfig,
    NON_TRAVERSABLE_EDGE_TYPES,
)
from graph_memory.retrieval.methods.epgm.relations import RelationAffinity


@dataclass(frozen=True)
class TypedArc:
    """One traversable step over a stored provenance edge.

    ``direction`` records whether the walk followed the stored orientation, so
    an emitted candidate edge can be oriented the way the graph recorded it even
    when the walk ran against that orientation.
    """

    source: str
    target: str
    edge_type: str
    direction: str  # "forward" | "reverse"


@dataclass(frozen=True)
class PartnerProposal:
    anchor_id: str
    partner_id: str
    path_node_ids: tuple[str, ...]
    arcs: tuple[TypedArc, ...]
    confidence: float

    @property
    def hops(self) -> int:
        return len(self.arcs)

    @property
    def connector_ids(self) -> tuple[str, ...]:
        """Interior nodes of the path; never ranked, always reported."""
        return self.path_node_ids[1:-1]


@dataclass(frozen=True)
class PromotionOutcome:
    proposal: PartnerProposal
    accepted: bool
    rejection_reason: str | None
    original_partner_rank: int
    final_partner_rank: int


@dataclass(frozen=True)
class CandidateDependency:
    source: str
    target: str
    edge_type: str
    confidence: float


def build_typed_adjacency(
    graph: ExecutionProvenanceGraph,
) -> Mapping[str, tuple[TypedArc, ...]]:
    """Undirected typed adjacency over traversable provenance edges.

    Traversal is undirected because the evidence partner is not reliably
    downstream of the anchor: real agent traces frequently record the supporting
    source upstream of the conclusion that a query asks about. Stored direction
    is retained on each arc rather than discarded, so orientation survives for
    edge emission.

    Zero-weight edges are explicit abstentions and are dropped. Edge weight
    magnitude is otherwise unused, which keeps behavior identical on graphs that
    store a constant placeholder weight.
    """

    adjacency: dict[str, list[TypedArc]] = defaultdict(list)
    for edge in graph.edges:
        edge_type = edge.edge_type.value
        if edge_type in NON_TRAVERSABLE_EDGE_TYPES:
            continue
        if float(edge.weight) <= 0.0:
            continue
        if edge.source == edge.target:
            continue
        adjacency[edge.source].append(
            TypedArc(edge.source, edge.target, edge_type, "forward")
        )
        adjacency[edge.target].append(
            TypedArc(edge.target, edge.source, edge_type, "reverse")
        )
    return {
        node_id: tuple(
            sorted(arcs, key=lambda arc: (arc.target, arc.edge_type, arc.direction))
        )
        for node_id, arcs in sorted(adjacency.items())
    }


def propose_partners(
    graph: ExecutionProvenanceGraph,
    *,
    anchor_ids: Sequence[str],
    candidate_ids: frozenset[str],
    dense_relevance: Mapping[str, float],
    relation_affinities: Sequence[RelationAffinity],
    config: EpgmRetrieverConfig,
) -> tuple[PartnerProposal, ...]:
    """Enumerate the best proposal per ``(anchor, partner)`` pair.

    Confidence is::

        dense_rel(partner) * prod(preference * specificity) * hop_decay^(hops-1)

    The partner's own Dense relevance is the leading term, and the anchor's is
    absent. That asymmetry is measured, not assumed: on the RQ3 trace, ranking
    proposals by partner relevance alone gives precision@10 0.60, while ranking
    by anchor relevance alone gives 0.00. The anchor's job is to decide *where*
    to look and *where to insert*; it says nothing about which of its neighbours
    is the missing evidence. Including it only flattened the score across every
    partner sharing one anchor and pushed the decision onto a tie-break.

    Relation *traversal weight* (query preference times graph specificity) is
    used rather than the affinity distribution; see :class:`RelationAffinity`
    for why both corrections are necessary. Adding specificity moves
    precision@10 from 0.60 to 0.80 on the same proposals.
    """

    weight_by_type = {
        item.edge_type: item.traversal_weight for item in relation_affinities
    }
    adjacency = build_typed_adjacency(graph)
    best: dict[tuple[str, str], PartnerProposal] = {}

    for anchor_id in anchor_ids:
        if dense_relevance.get(anchor_id, 0.0) <= 0.0:
            # A zero-relevance anchor is not a credible place to start looking.
            continue
        # Breadth-first so the shortest path to each partner is seen first.
        queue: deque[tuple[str, tuple[str, ...], tuple[TypedArc, ...], float]] = deque()
        queue.append((anchor_id, (anchor_id,), (), 1.0))
        while queue:
            node_id, path_nodes, arcs, weight_product = queue.popleft()
            if len(arcs) >= config.max_hops:
                continue
            for arc in adjacency.get(node_id, ()):
                if arc.target in path_nodes:
                    continue
                weight = weight_by_type.get(arc.edge_type)
                if weight is None or weight <= 0.0:
                    continue
                next_nodes = (*path_nodes, arc.target)
                next_arcs = (*arcs, arc)
                next_product = weight_product * weight
                if arc.target in candidate_ids and arc.target != anchor_id:
                    confidence = (
                        dense_relevance.get(arc.target, 0.0)
                        * next_product
                        * config.hop_decay ** (len(next_arcs) - 1)
                    )
                    key = (anchor_id, arc.target)
                    current = best.get(key)
                    if current is None or (
                        confidence,
                        tuple(reversed(next_nodes)),
                    ) > (current.confidence, tuple(reversed(current.path_node_ids))):
                        best[key] = PartnerProposal(
                            anchor_id=anchor_id,
                            partner_id=arc.target,
                            path_node_ids=next_nodes,
                            arcs=next_arcs,
                            confidence=confidence,
                        )
                queue.append((arc.target, next_nodes, next_arcs, next_product))

    return tuple(
        sorted(
            best.values(),
            key=lambda item: (
                -item.confidence,
                item.anchor_id,
                item.partner_id,
                item.path_node_ids,
            ),
        )
    )


def select_promotions(
    proposals: Sequence[PartnerProposal],
    *,
    dense_rank: Mapping[str, int],
    config: EpgmRetrieverConfig,
) -> dict[PartnerProposal, tuple[bool, str | None]]:
    """One partner per anchor and one anchor per partner, with reasons.

    One-to-one matching is what keeps precision usable. Without it a single
    strong anchor floods the ranked head with its entire neighbourhood, which
    inflates emitted-edge count while lowering edge precision.
    """

    outcomes: dict[PartnerProposal, tuple[bool, str | None]] = {}
    eligible_by_anchor: dict[str, list[PartnerProposal]] = defaultdict(list)
    for proposal in proposals:
        partner_rank = dense_rank[proposal.partner_id]
        anchor_rank = dense_rank[proposal.anchor_id]
        if proposal.confidence < config.min_partner_confidence:
            outcomes[proposal] = (False, "below_partner_confidence")
        elif partner_rank <= config.preserve_dense_top_n:
            outcomes[proposal] = (False, "protected_partner")
        elif partner_rank <= anchor_rank:
            outcomes[proposal] = (False, "partner_not_after_anchor")
        else:
            eligible_by_anchor[proposal.anchor_id].append(proposal)

    anchor_winners: list[PartnerProposal] = []
    for anchor_id in sorted(eligible_by_anchor):
        ordered = sorted(
            eligible_by_anchor[anchor_id],
            key=lambda item: (
                -item.confidence,
                dense_rank[item.partner_id],
                item.partner_id,
                item.path_node_ids,
            ),
        )
        anchor_winners.append(ordered[0])
        for proposal in ordered[1:]:
            outcomes[proposal] = (False, "lower_confidence_for_anchor")

    by_partner: dict[str, list[PartnerProposal]] = defaultdict(list)
    for proposal in anchor_winners:
        by_partner[proposal.partner_id].append(proposal)
    for partner_id in sorted(by_partner):
        ordered = sorted(
            by_partner[partner_id],
            key=lambda item: (
                -item.confidence,
                dense_rank[item.anchor_id],
                item.anchor_id,
                item.path_node_ids,
            ),
        )
        outcomes[ordered[0]] = (True, None)
        for proposal in ordered[1:]:
            outcomes[proposal] = (False, "partner_conflict")
    return outcomes


def apply_promotions(
    dense_ids: Sequence[str],
    *,
    outcomes: dict[PartnerProposal, tuple[bool, str | None]],
    dense_rank: Mapping[str, int],
    config: EpgmRetrieverConfig,
) -> tuple[list[str], tuple[PartnerProposal, ...]]:
    """Move each accepted partner to just after its anchor.

    Only ids move; the caller keeps the ordered Dense score slots. The result is
    therefore a pure permutation that preserves the Dense score multiset exactly
    and cannot inflate any score-based metric.

    Unlike a membership-only selector, this may reorder inside the ranked head.
    That is deliberate: on the RQ2 test split 27.4% of tasks already hold every
    gold node inside the Dense top-10 but not inside the top-5, so intra-head
    reordering is where the available headroom actually is.
    """

    final_ids = list(dense_ids)
    promoted: list[PartnerProposal] = []
    accepted = sorted(
        (item for item, outcome in outcomes.items() if outcome[0]),
        key=lambda item: (
            dense_rank[item.anchor_id],
            dense_rank[item.partner_id],
            item.partner_id,
        ),
    )
    for proposal in accepted:
        anchor_position = final_ids.index(proposal.anchor_id)
        partner_position = final_ids.index(proposal.partner_id)
        insertion_position = max(config.preserve_dense_top_n, anchor_position + 1)
        if partner_position <= insertion_position:
            outcomes[proposal] = (False, "no_effective_insertion")
            continue
        final_ids.pop(partner_position)
        final_ids.insert(insertion_position, proposal.partner_id)
        promoted.append(proposal)
    return final_ids, tuple(promoted)


def candidate_dependency(proposal: PartnerProposal) -> CandidateDependency | None:
    """Collapse a promoted path into one oriented candidate dependency.

    Returns ``None`` when the path establishes no directed dependency between
    its endpoints: either it carries no contracted dependency relation, or its
    steps disagree on direction. ``A <- X -> B`` is genuinely connected but
    implies neither ``A -> B`` nor ``B -> A``, so emitting an edge there would
    fabricate a dependency the graph never recorded.
    """

    if not any(arc.edge_type in EMITTABLE_EDGE_TYPES for arc in proposal.arcs):
        return None
    directions = {arc.direction for arc in proposal.arcs}
    if len(directions) != 1:
        return None
    source, target = proposal.anchor_id, proposal.partner_id
    if directions == {"reverse"}:
        source, target = target, source
    if source == target:
        return None
    emittable = [
        arc.edge_type
        for arc in proposal.arcs
        if arc.edge_type in EMITTABLE_EDGE_TYPES
    ]
    edge_type = emittable[0] if len(emittable) == 1 else "depends_on"
    return CandidateDependency(
        source=source,
        target=target,
        edge_type=edge_type,
        confidence=proposal.confidence,
    )


__all__ = [
    "CandidateDependency",
    "PartnerProposal",
    "PromotionOutcome",
    "TypedArc",
    "apply_promotions",
    "build_typed_adjacency",
    "candidate_dependency",
    "propose_partners",
    "select_promotions",
]
