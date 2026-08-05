from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from graph_memory.graphs.contracts import GraphEdge
from graph_memory.graphs.provenance import PRECEDES_EDGE, ProvenanceEdge
from graph_memory.retrieval.contracts import (
    CandidateEdgeTrace,
    DenseRankTrace,
    ProvenanceGraphEdgeTrace,
    ProvenancePathProposalTrace,
    ProvenancePathTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class ProvenancePathConfig:
    seed_top_s: int = 5
    max_path_hops: int = 6
    max_partners_per_anchor: int = 3
    max_expansions: int = 256
    preserve_dense_top_n: int = 2

    def __post_init__(self) -> None:
        for name in (
            "seed_top_s",
            "max_path_hops",
            "max_partners_per_anchor",
            "max_expansions",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.preserve_dense_top_n < 0:
            raise ValueError("preserve_dense_top_n must be non-negative")


@dataclass(frozen=True)
class _PathEdge:
    edge_id: str
    relation: str
    source: str
    target: str


@dataclass(frozen=True)
class _Path:
    node_ids: tuple[str, ...]
    edges: tuple[_PathEdge, ...]
    traversed_reverse: tuple[bool, ...]


@dataclass(frozen=True)
class _Proposal:
    anchor_id: str
    partner_id: str
    path: _Path


@dataclass(frozen=True)
class ProvenancePathMethod:
    dense_ranker: DenseTaskRetriever
    config: ProvenancePathConfig = ProvenancePathConfig()
    name: str = "provenance_path"

    def rank_task(
        self,
        request: RankingMethodRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        if not isinstance(request, ExecutionProvenanceRankingRequest):
            raise TypeError(
                f"{self.name} requires ExecutionProvenanceRankingRequest, "
                f"got {type(request).__name__}."
            )
        text_request = TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
        )
        dense_ranked = self.dense_ranker.rank(text_request)
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        seed_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        proposals = _discover_proposals(
            seed_ids,
            request.graph.edges,
            dense_rank=dense_rank,
            config=self.config,
        )
        outcomes = _select_proposals(
            proposals,
            dense_rank=dense_rank,
            config=self.config,
        )
        final_ids, moved = _apply_stable_insertions(
            [node.node_id for node in dense_ranked],
            proposals=proposals,
            outcomes=outcomes,
            dense_rank=dense_rank,
            config=self.config,
        )
        score_slots = [node.score for node in dense_ranked]
        ranked_nodes = (
            dense_ranked
            if not moved
            else [
                RankedNode(node_id=node_id, score=score_slots[index])
                for index, node_id in enumerate(final_ids)
            ]
        )
        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }
        top_ids = set(final_ids[:top_k])
        emitted_proposals = tuple(
            proposal
            for proposal in moved
            if proposal.anchor_id in top_ids and proposal.partner_id in top_ids
        )
        edges = tuple(
            GraphEdge(
                source=proposal.anchor_id,
                target=proposal.partner_id,
                edge_type="provenance_path",
                weight=1.0,
                directed=False,
            )
            for proposal in emitted_proposals
        )
        traced_edges = _traced_graph_edges(proposals)
        trace = ProvenancePathTrace(
            dense_ranks=tuple(
                DenseRankTrace(
                    node_id=node.node_id,
                    dense_rank=dense_rank[node.node_id],
                    dense_score=node.score,
                    final_rank=final_rank[node.node_id],
                )
                for node in dense_ranked
            ),
            seed_candidate_ids=seed_ids,
            graph_edges=tuple(
                ProvenanceGraphEdgeTrace(
                    edge_id=edge.edge_id,
                    source_node_id=edge.source,
                    target_node_id=edge.target,
                    relation=edge.relation,
                )
                for edge in traced_edges
            ),
            proposals=tuple(
                ProvenancePathProposalTrace(
                    anchor_candidate_id=proposal.anchor_id,
                    partner_candidate_id=proposal.partner_id,
                    path_node_ids=proposal.path.node_ids,
                    path_relations=tuple(edge.relation for edge in proposal.path.edges),
                    path_edge_ids=tuple(edge.edge_id for edge in proposal.path.edges),
                    traversed_reverse=proposal.path.traversed_reverse,
                    accepted=outcomes[proposal] is None,
                    rejection_reason=outcomes[proposal],
                    original_anchor_rank=dense_rank[proposal.anchor_id],
                    original_partner_rank=dense_rank[proposal.partner_id],
                    final_partner_rank=final_rank[proposal.partner_id],
                )
                for proposal in proposals
            ),
            protected_prefix=tuple(
                node.node_id
                for node in dense_ranked[: self.config.preserve_dense_top_n]
            ),
            emitted_edges=tuple(
                CandidateEdgeTrace(
                    source=proposal.anchor_id,
                    target=proposal.partner_id,
                    edge_type="provenance_path",
                    confidence=1.0,
                )
                for proposal in emitted_proposals
            ),
            seed_top_s=self.config.seed_top_s,
            max_path_hops=self.config.max_path_hops,
            max_partners_per_anchor=self.config.max_partners_per_anchor,
            max_expansions=self.config.max_expansions,
            exact_dense_fallback=not moved,
        )
        return RetrievalMethodResult(
            ranked_nodes=tuple(ranked_nodes),
            trace=RetrievalTrace(retrieved_edges=edges, native_trace=trace),
        )


def _discover_proposals(
    seed_ids: tuple[str, ...],
    graph_edges: tuple[ProvenanceEdge, ...],
    *,
    dense_rank: dict[str, int],
    config: ProvenancePathConfig,
) -> tuple[_Proposal, ...]:
    adjacency: dict[str, list[tuple[str, _PathEdge, bool]]] = defaultdict(list)
    for edge in graph_edges:
        if edge.relation == PRECEDES_EDGE:
            continue
        path_edge = _PathEdge(
            edge_id=edge.edge_id,
            relation=edge.relation,
            source=edge.source,
            target=edge.target,
        )
        adjacency[edge.source].append((edge.target, path_edge, False))
        adjacency[edge.target].append((edge.source, path_edge, True))
    for values in adjacency.values():
        values.sort(
            key=lambda item: (
                item[1].relation,
                item[2],
                dense_rank.get(item[0], len(dense_rank) + 1),
                item[0],
            )
        )

    proposals: list[_Proposal] = []
    expansions = 0
    for anchor_id in seed_ids:
        queue: deque[_Path] = deque([_Path((anchor_id,), (), ())])
        best_by_partner: dict[str, _Path] = {}
        while queue and expansions < config.max_expansions:
            path = queue.popleft()
            if len(path.edges) >= config.max_path_hops:
                continue
            current = path.node_ids[-1]
            for partner_id, edge, reverse in adjacency.get(current, ()):
                if expansions >= config.max_expansions:
                    break
                expansions += 1
                if partner_id in path.node_ids:
                    continue
                extended = _Path(
                    node_ids=(*path.node_ids, partner_id),
                    edges=(*path.edges, edge),
                    traversed_reverse=(*path.traversed_reverse, reverse),
                )
                if partner_id in dense_rank and partner_id != anchor_id:
                    previous = best_by_partner.get(partner_id)
                    if previous is None or _path_key(extended, dense_rank) < _path_key(
                        previous, dense_rank
                    ):
                        best_by_partner[partner_id] = extended
                queue.append(extended)
        proposals.extend(
            _Proposal(anchor_id=anchor_id, partner_id=partner_id, path=path)
            for partner_id, path in sorted(
                best_by_partner.items(),
                key=lambda item: _path_key(item[1], dense_rank),
            )
        )
    return tuple(
        sorted(
            proposals,
            key=lambda proposal: (
                dense_rank[proposal.anchor_id],
                _path_key(proposal.path, dense_rank),
            ),
        )
    )


def _path_key(path: _Path, dense_rank: dict[str, int]) -> tuple[object, ...]:
    return (
        len(path.edges),
        tuple(edge.relation for edge in path.edges),
        path.traversed_reverse,
        dense_rank.get(path.node_ids[-1], len(dense_rank) + 1),
        path.node_ids[-1],
    )


def _select_proposals(
    proposals: tuple[_Proposal, ...],
    *,
    dense_rank: dict[str, int],
    config: ProvenancePathConfig,
) -> dict[_Proposal, str | None]:
    outcomes: dict[_Proposal, str | None] = {}
    accepted_per_anchor: dict[str, int] = defaultdict(int)
    claimed_partner: set[str] = set()
    for proposal in proposals:
        partner_rank = dense_rank[proposal.partner_id]
        anchor_rank = dense_rank[proposal.anchor_id]
        if partner_rank <= config.preserve_dense_top_n:
            outcomes[proposal] = "protected_partner"
        elif partner_rank <= anchor_rank:
            outcomes[proposal] = "partner_not_after_anchor"
        elif accepted_per_anchor[proposal.anchor_id] >= config.max_partners_per_anchor:
            outcomes[proposal] = "anchor_partner_limit"
        elif proposal.partner_id in claimed_partner:
            outcomes[proposal] = "partner_conflict"
        else:
            outcomes[proposal] = None
            accepted_per_anchor[proposal.anchor_id] += 1
            claimed_partner.add(proposal.partner_id)
    return outcomes


def _apply_stable_insertions(
    dense_ids: list[str],
    *,
    proposals: tuple[_Proposal, ...],
    outcomes: dict[_Proposal, str | None],
    dense_rank: dict[str, int],
    config: ProvenancePathConfig,
) -> tuple[list[str], tuple[_Proposal, ...]]:
    final_ids = list(dense_ids)
    moved: list[_Proposal] = []
    inserted_per_anchor: dict[str, int] = defaultdict(int)
    for proposal in sorted(
        (proposal for proposal in proposals if outcomes[proposal] is None),
        key=lambda item: (
            dense_rank[item.anchor_id],
            _path_key(item.path, dense_rank),
        ),
    ):
        anchor_position = final_ids.index(proposal.anchor_id)
        partner_position = final_ids.index(proposal.partner_id)
        insertion_position = max(
            config.preserve_dense_top_n,
            anchor_position + 1 + inserted_per_anchor[proposal.anchor_id],
        )
        if partner_position <= insertion_position:
            outcomes[proposal] = "no_effective_insertion"
            continue
        final_ids.pop(partner_position)
        final_ids.insert(insertion_position, proposal.partner_id)
        inserted_per_anchor[proposal.anchor_id] += 1
        moved.append(proposal)
    return final_ids, tuple(moved)


def _traced_graph_edges(proposals: tuple[_Proposal, ...]) -> tuple[_PathEdge, ...]:
    unique = {
        edge.edge_id: edge
        for proposal in proposals
        for edge in proposal.path.edges
    }
    return tuple(unique[edge_id] for edge_id in sorted(unique))


__all__ = ["ProvenancePathConfig", "ProvenancePathMethod"]
