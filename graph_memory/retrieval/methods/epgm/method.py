"""The single non-trained EPGM provenance retriever: typed partner completion.

From each leading Dense anchor, walk query-relevant typed provenance edges a
bounded number of hops, accept at most one evidence partner per anchor, and
promote it to just after its anchor. Frozen query-to-relation affinity supplies
the typed signal; there is no diffusion, no budgeted selection, and no
hand-authored schema gate, so one configuration serves both synthetic dependency
graphs and real agent traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.graphs.provenance import (
    ExecutionProvenanceEdge,
    ProvenanceEdgeType,
)
from graph_memory.retrieval.contracts import (
    CandidateEdgeTrace,
    DenseRankTrace,
    ProvenanceBindingTrace,
    ProvenanceEdgeTrace,
    ProvenanceRelationAffinityTrace,
    PartnerProposalTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
    TypedPartnerCompletionTrace,
)
from graph_memory.retrieval.methods.epgm.config import (
    EpgmRetrieverConfig,
    NON_TRAVERSABLE_EDGE_TYPES,
)
from graph_memory.retrieval.methods.epgm.relations import (
    encode_relation_vectors,
    query_relation_affinities,
)
from graph_memory.retrieval.methods.epgm.walk import (
    PartnerProposal,
    apply_promotions,
    candidate_dependency,
    propose_partners,
    select_promotions,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class EpgmRetriever:
    """Typed partner completion over an execution provenance graph."""

    dense_ranker: DenseTaskRetriever
    config: EpgmRetrieverConfig = EpgmRetrieverConfig()
    name: str = "execution_provenance_retriever"
    display_name: str = "EPGM (non-trained)"
    _relation_vectors: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False, compare=False
    )

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
        dense_ranked, query_vector = self.dense_ranker.rank_with_query_vector(
            TextRankingRequest(request.task_id, request.query_text, request.candidates)
        )
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        dense_relevance = _normalized_relevance(dense_ranked)
        traversable = _traversable_edge_types(request)
        relations = query_relation_affinities(
            query_vector,
            self._cached_relation_vectors(),
            traversable,
            self.config,
            edge_type_counts=_edge_type_counts(request, traversable),
        )
        anchor_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.anchor_top_a]
        )
        proposals = propose_partners(
            request.graph,
            anchor_ids=anchor_ids,
            candidate_ids=frozenset(dense_rank),
            dense_relevance=dense_relevance,
            relation_affinities=relations,
            config=self.config,
        )
        outcomes = select_promotions(
            proposals, dense_rank=dense_rank, config=self.config
        )
        final_ids, promoted = apply_promotions(
            [node.node_id for node in dense_ranked],
            outcomes=outcomes,
            dense_rank=dense_rank,
            config=self.config,
        )
        if promoted:
            # Only ids move between slots; the ordered Dense score sequence is
            # reused verbatim so the returned score multiset is unchanged.
            score_slots = [node.score for node in dense_ranked]
            ranked_nodes = [
                RankedNode(node_id, score_slots[index])
                for index, node_id in enumerate(final_ids)
            ]
        else:
            ranked_nodes = dense_ranked
        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }

        top_ids = {node.node_id for node in ranked_nodes[:top_k]}
        dependencies = [
            dependency
            for proposal in promoted
            if proposal.anchor_id in top_ids and proposal.partner_id in top_ids
            if (dependency := candidate_dependency(proposal)) is not None
        ]
        seen: set[tuple[str, str]] = set()
        emitted = []
        for dependency in dependencies:
            key = (dependency.source, dependency.target)
            if key in seen:
                continue
            seen.add(key)
            emitted.append(dependency)
        connector_ids = tuple(
            sorted(
                {
                    node_id
                    for proposal in promoted
                    for node_id in proposal.connector_ids
                    if node_id not in dense_rank
                }
            )
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=[
                    GraphEdge(
                        source=dependency.source,
                        target=dependency.target,
                        edge_type="feeds",
                        weight=dependency.confidence,
                        directed=True,
                    )
                    for dependency in emitted
                ],
                native_trace=TypedPartnerCompletionTrace(
                    dense_ranks=tuple(
                        DenseRankTrace(
                            node_id=node.node_id,
                            dense_rank=dense_rank[node.node_id],
                            dense_score=node.score,
                            final_rank=final_rank[node.node_id],
                        )
                        for node in dense_ranked
                    ),
                    anchor_candidate_ids=anchor_ids,
                    relation_description_version=(
                        self.config.relation_description_version
                    ),
                    relations=tuple(
                        ProvenanceRelationAffinityTrace(
                            item.edge_type, item.similarity, item.affinity
                        )
                        for item in relations
                    ),
                    proposals=tuple(
                        PartnerProposalTrace(
                            anchor_id=proposal.anchor_id,
                            partner_id=proposal.partner_id,
                            path_node_ids=proposal.path_node_ids,
                            path_edge_types=tuple(
                                arc.edge_type for arc in proposal.arcs
                            ),
                            path_directions=tuple(
                                cast(Literal["forward", "reverse"], arc.direction)
                                for arc in proposal.arcs
                            ),
                            confidence=proposal.confidence,
                            accepted=outcomes[proposal][0],
                            rejection_reason=outcomes[proposal][1],
                            original_partner_rank=dense_rank[proposal.partner_id],
                            final_partner_rank=final_rank[proposal.partner_id],
                        )
                        for proposal in _ordered_proposals(proposals, dense_rank)
                    ),
                    protected_prefix=tuple(
                        node.node_id
                        for node in dense_ranked[: self.config.preserve_dense_top_n]
                    ),
                    connector_node_ids=connector_ids,
                    promoted_native_edges=_promoted_native_edge_traces(
                        promoted, request
                    ),
                    exact_dense_fallback=not promoted,
                    emitted_edges=tuple(
                        CandidateEdgeTrace(
                            dependency.source,
                            dependency.target,
                            dependency.edge_type,
                            dependency.confidence,
                        )
                        for dependency in emitted
                    ),
                    scorer_identity=_scorer_identity(request),
                ),
            ),
        )

    def _cached_relation_vectors(self) -> NDArray[np.float64]:
        """Relation descriptions are frozen, so encode them at most once."""

        relation_vectors = self._relation_vectors
        if relation_vectors is None:
            relation_vectors = encode_relation_vectors(
                self.dense_ranker.encoder,
                self.config,
                passage_prefix=self.dense_ranker.config.passage_prefix,
                batch_size=self.dense_ranker.config.batch_size,
            )
            object.__setattr__(self, "_relation_vectors", relation_vectors)
        return relation_vectors


def _edge_type_counts(
    request: ExecutionProvenanceRankingRequest,
    traversable: frozenset[str],
) -> dict[str, int]:
    """How often each traversable relation occurs in this graph.

    Feeds the inverse-frequency weight, so relation specificity is derived from
    the graph in front of the retriever rather than from a corpus constant.
    """

    counts: dict[str, int] = dict.fromkeys(traversable, 0)
    for edge in request.graph.edges:
        edge_type = edge.edge_type.value
        if edge_type in counts and float(edge.weight) > 0.0:
            counts[edge_type] += 1
    return counts


def _traversable_edge_types(
    request: ExecutionProvenanceRankingRequest,
) -> frozenset[str]:
    """Present edge types the walk may actually use.

    Non-traversable relations are excluded before the affinity softmax so the
    distribution describes the choice the walk really faces. Including them
    would let an untraversable relation absorb probability mass and rescale
    every traversable preference.
    """

    return frozenset(
        edge.edge_type.value
        for edge in request.graph.edges
        if edge.edge_type.value not in NON_TRAVERSABLE_EDGE_TYPES
        if float(edge.weight) > 0.0
    )


def _normalized_relevance(dense_ranked: list[RankedNode]) -> dict[str, float]:
    """Dense relevance rescaled against the best candidate for this query.

    Peak normalization, not min-max. Min-max is degenerate here in two ways that
    both silently disable promotion: it sends the lowest-ranked candidate to
    exactly 0.0, and when several candidates tie at the minimum it sends all of
    them there. Since confidence multiplies by partner relevance, a 0.0 partner
    can never be promoted no matter how strong its structural evidence is.

    Dividing by the peak keeps the value scale-free and comparable across
    queries, stays monotone in the raw score, and reads naturally: 0.5 means
    half as query-relevant as the best candidate. Encoder scores are cosine
    similarities, so negatives are clipped to zero rather than reflected.
    """

    if not dense_ranked:
        return {}
    peak = max(node.score for node in dense_ranked)
    if peak <= 0.0:
        return {node.node_id: 1.0 for node in dense_ranked}
    return {
        node.node_id: max(node.score, 0.0) / peak for node in dense_ranked
    }


def _ordered_proposals(
    proposals: tuple[PartnerProposal, ...],
    dense_rank: dict[str, int],
) -> tuple[PartnerProposal, ...]:
    return tuple(
        sorted(
            proposals,
            key=lambda item: (
                dense_rank[item.anchor_id],
                -item.confidence,
                dense_rank[item.partner_id],
                item.partner_id,
                item.path_node_ids,
            ),
        )
    )


def _promoted_native_edge_traces(
    promoted: tuple[PartnerProposal, ...],
    request: ExecutionProvenanceRankingRequest,
) -> tuple[ProvenanceEdgeTrace, ...]:
    """Stored provenance edges underlying every accepted promotion path."""

    edge_by_key = {
        (edge.source, edge.target, edge.edge_type.value): edge
        for edge in request.graph.edges
    }
    keys = {
        (
            arc.source if arc.direction == "forward" else arc.target,
            arc.target if arc.direction == "forward" else arc.source,
            arc.edge_type,
        )
        for proposal in promoted
        for arc in proposal.arcs
    }
    missing = keys - edge_by_key.keys()
    if missing:
        raise ValueError(
            "promoted arc does not map to stored provenance edges: "
            f"missing={sorted(missing)}"
        )
    return tuple(_edge_trace(edge_by_key[key]) for key in sorted(keys))


def _edge_trace(edge: ExecutionProvenanceEdge) -> ProvenanceEdgeTrace:
    semantic_rank = edge.metadata.get("semantic_rank")
    semantic_score = edge.metadata.get("semantic_score")
    return ProvenanceEdgeTrace(
        source=edge.source,
        target=edge.target,
        edge_type=edge.edge_type,
        weight=edge.weight,
        binding=(
            ProvenanceBindingTrace(
                output_field=edge.binding.output_field,
                input_parameter=edge.binding.input_parameter,
                binding_kind=edge.binding.binding_kind,
            )
            if edge.binding is not None
            else None
        ),
        semantic_rank=(
            cast(int, semantic_rank)
            if isinstance(semantic_rank, int) and not isinstance(semantic_rank, bool)
            else None
        ),
        semantic_score=(
            float(semantic_score)
            if isinstance(semantic_score, (int, float))
            and not isinstance(semantic_score, bool)
            else None
        ),
    )


def _scorer_identity(request: ExecutionProvenanceRankingRequest) -> str:
    identities = {
        value
        for edge in request.graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
        if isinstance((value := edge.metadata.get("semantic_scorer")), str)
    }
    return ",".join(sorted(identities)) or "unknown"


__all__ = ["EpgmRetriever"]
