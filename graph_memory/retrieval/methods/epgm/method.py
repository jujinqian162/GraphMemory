"""The single non-trained EPGM provenance retriever.

``ppr_steiner`` is the reported default: frozen query/relation semantics define
source-local typed transitions, Personalized PageRank expands over the native
graph, and a budgeted connected selector returns auditable evidence. Historical
``typed_beam`` and ``dependency_path`` variants remain diagnostics under the
same registry id.
"""

from __future__ import annotations

from collections import defaultdict
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
    ProvenanceCandidatePrizeTrace,
    ProvenanceEdgeTrace,
    ProvenancePprNodeTrace,
    ProvenanceRelationAffinityTrace,
    ProvenanceSelectedArcTrace,
    ProvenanceSelectionStepTrace,
    ProvenanceTransitionTrace,
    QueryConditionedExecutionProvenanceTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
    StatelessExecutionProvenanceTrace,
    StatelessProvenancePathTrace,
)
from graph_memory.retrieval.methods.epgm.config import EpgmRetrieverConfig
from graph_memory.retrieval.methods.epgm.diffusion import (
    dense_teleport,
    encode_relation_vectors,
    TypedTransition,
    personalized_pagerank,
    query_relation_affinities,
)
from graph_memory.retrieval.methods.epgm.search import EpgmPath, search_epgm_paths
from graph_memory.retrieval.methods.epgm.selection import select_budgeted_subgraph
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class EpgmRankedNode:
    node_id: str
    score: float
    dense_rank: int
    dense_score: float
    graph_score: float
    best_path: EpgmPath | None


@dataclass(frozen=True)
class EpgmRetrievalResult:
    ranked_nodes: tuple[EpgmRankedNode, ...]
    seed_ids: tuple[str, ...]
    paths: tuple[EpgmPath, ...]


@dataclass(frozen=True)
class EpgmRetriever:
    """One non-trained EPGM implementation with explicit diagnostics."""

    dense_ranker: DenseTaskRetriever
    config: EpgmRetrieverConfig = EpgmRetrieverConfig()
    name: str = "execution_provenance_retriever"
    display_name: str = "EPGM (non-trained)"
    _relation_vectors: NDArray[np.float64] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def variant(self) -> str:
        return self.config.variant

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
        if self.config.variant == "ppr_steiner":
            return self._rank_connected_subgraph(request, top_k=top_k)
        dense_ranked = self.dense_ranker.rank(
            TextRankingRequest(request.task_id, request.query_text, request.candidates)
        )
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        seed_relevance = _normalized_relevance(dense_ranked)
        seed_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        paths = search_epgm_paths(
            request.graph,
            seed_ids,
            seed_relevance,
            candidate_ids=frozenset(dense_rank),
            config=self.config,
        )
        best_by_target = _best_path_by_target(
            paths, min_path_confidence=self.config.min_path_confidence
        )
        outcomes = _select_paths(paths, dense_rank=dense_rank, config=self.config)

        if self.config.fusion == "stable_insert":
            ranked_nodes, promoted = _stable_insert_ranking(
                dense_ranked,
                outcomes=outcomes,
                dense_rank=dense_rank,
                preserve_dense_top_n=self.config.preserve_dense_top_n,
            )
            exact_dense_fallback = not promoted
        else:
            ranked_nodes = _additive_ranking(
                dense_ranked,
                seed_relevance=seed_relevance,
                best_by_target=best_by_target,
                graph_weight=self.config.graph_weight,
            )
            promoted = tuple(
                path for path, (accepted, _reason) in outcomes.items() if accepted
            )
            exact_dense_fallback = not promoted

        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }
        top_ids = {node.node_id for node in ranked_nodes[:top_k]}
        emitted = tuple(
            path
            for path in promoted
            if path.seed_id in top_ids and path.target_id in top_ids
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=_logical_edges(emitted),
                native_trace=StatelessExecutionProvenanceTrace(
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
                    paths=tuple(
                        StatelessProvenancePathTrace(
                            anchor_id=path.seed_id,
                            partner_id=path.target_id,
                            node_ids=path.node_ids,
                            path_confidence=path.score,
                            binding_valid=path.gate.binding_valid,
                            completeness_valid=path.gate.completeness_valid,
                            lifecycle_valid=path.gate.lifecycle_valid,
                            accepted=outcomes[path][0],
                            rejection_reason=outcomes[path][1],
                            original_partner_rank=dense_rank[path.target_id],
                            final_partner_rank=final_rank[path.target_id],
                        )
                        for path in paths
                    ),
                    edges=_edge_traces(paths, request),
                    protected_prefix=tuple(
                        node.node_id
                        for node in dense_ranked[: self.config.preserve_dense_top_n]
                    ),
                    exact_dense_fallback=exact_dense_fallback,
                    emitted_edges=tuple(
                        CandidateEdgeTrace(
                            source=_logical_endpoints(path)[0],
                            target=_logical_endpoints(path)[1],
                            edge_type=_logical_edge_type(path),
                            confidence=path.score,
                        )
                        for path in emitted
                    ),
                    scorer_identity=_scorer_identity(request),
                    variant=self.config.variant,
                ),
            ),
        )

    def _rank_connected_subgraph(
        self,
        request: ExecutionProvenanceRankingRequest,
        *,
        top_k: int,
    ) -> RetrievalMethodResult:
        dense_ranked, query_vector = self.dense_ranker.rank_with_query_vector(
            TextRankingRequest(request.task_id, request.query_text, request.candidates)
        )
        dense_scores = {node.node_id: node.score for node in dense_ranked}
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        teleport = dense_teleport(dense_scores, request.graph)
        relation_vectors = self._relation_vectors
        if relation_vectors is None:
            relation_vectors = encode_relation_vectors(
                self.dense_ranker.encoder,
                self.config,
                passage_prefix=self.dense_ranker.config.passage_prefix,
                batch_size=self.dense_ranker.config.batch_size,
            )
            object.__setattr__(self, "_relation_vectors", relation_vectors)
        present_types = frozenset(edge.edge_type.value for edge in request.graph.edges)
        relations = query_relation_affinities(
            query_vector,
            relation_vectors,
            present_types,
            self.config,
        )
        ppr = personalized_pagerank(request.graph, teleport, relations, self.config)
        selected, prizes = select_budgeted_subgraph(
            dense_scores, ppr, top_k=top_k, config=self.config
        )

        if selected.exact_dense_fallback:
            ranked_nodes = dense_ranked
        else:
            # The selector chooses top-k membership against the Dense incumbent;
            # it does not receive permission to reorder the entire prefix by
            # graph centrality. Reconstruct the accepted membership changes and
            # retain Dense-relative order within both the top-k and tail. This
            # lets structure complete a support set without destroying early
            # semantic hits or MRR.
            evidence_budget = min(top_k, len(dense_ranked))
            incumbent_ids = {
                node.node_id for node in dense_ranked[:evidence_budget]
            }
            for step in selected.steps:
                incumbent_ids.difference_update(step.displaced_candidate_ids)
                incumbent_ids.update(step.added_candidate_ids)
            top_ids = [
                node.node_id for node in dense_ranked if node.node_id in incumbent_ids
            ]
            tail_ids = [
                node.node_id for node in dense_ranked if node.node_id not in incumbent_ids
            ]
            final_ids = [*top_ids, *tail_ids]
            score_slots = [node.score for node in dense_ranked]
            ranked_nodes = [
                RankedNode(node_id, score_slots[index])
                for index, node_id in enumerate(final_ids)
            ]
        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }
        retrieved_edges = [
            GraphEdge(
                source=edge.source,
                target=edge.target,
                edge_type="feeds",
                weight=edge.confidence,
                directed=True,
            )
            for edge in selected.candidate_edges
        ]
        selected_native_edges = _selected_native_edge_traces(
            selected.transitions, request
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=retrieved_edges,
                native_trace=QueryConditionedExecutionProvenanceTrace(
                    native_graph_node_ids=tuple(
                        sorted(node.node_id for node in request.graph.nodes)
                    ),
                    dense_ranks=tuple(
                        DenseRankTrace(
                            node_id=node.node_id,
                            dense_rank=dense_rank[node.node_id],
                            dense_score=node.score,
                            final_rank=final_rank[node.node_id],
                        )
                        for node in dense_ranked
                    ),
                    relation_description_version=(
                        self.config.relation_description_version
                    ),
                    relations=tuple(
                        ProvenanceRelationAffinityTrace(
                            item.edge_type, item.similarity, item.affinity
                        )
                        for item in ppr.relation_affinities
                    ),
                    transitions=tuple(
                        ProvenanceTransitionTrace(
                            source=item.source,
                            target=item.target,
                            edge_type=item.edge_type,
                            direction=cast(
                                Literal["forward", "reverse"], item.direction
                            ),
                            recorded_weight=item.recorded_weight,
                            relation_affinity=item.relation_affinity,
                            probability=item.probability,
                            cost=item.cost,
                        )
                        for item in ppr.transitions
                    ),
                    ppr_nodes=tuple(
                        ProvenancePprNodeTrace(
                            node_id=node_id,
                            teleport=ppr.teleport[node_id],
                            score=ppr.node_scores[node_id],
                        )
                        for node_id in sorted(ppr.node_scores)
                    ),
                    ppr_iterations=ppr.iterations,
                    ppr_residual=ppr.residual,
                    ppr_converged=ppr.converged,
                    candidate_prizes=tuple(
                        ProvenanceCandidatePrizeTrace(
                            item.node_id,
                            item.dense_component,
                            item.ppr_component,
                            item.prize,
                        )
                        for item in prizes
                    ),
                    selected_candidate_ids=selected.candidate_ids,
                    connector_node_ids=selected.connector_ids,
                    selection_steps=tuple(
                        ProvenanceSelectionStepTrace(
                            item.anchor_id,
                            item.target_id,
                            item.path_node_ids,
                            tuple(
                                ProvenanceSelectedArcTrace(
                                    transition.source,
                                    transition.target,
                                    transition.edge_type,
                                    cast(
                                        Literal["forward", "reverse"],
                                        transition.direction,
                                    ),
                                )
                                for transition in item.transitions
                            ),
                            item.added_candidate_ids,
                            item.displaced_candidate_ids,
                            item.prize_gain,
                            item.edge_cost,
                            item.displacement_cost,
                            item.marginal_gain,
                        )
                        for item in selected.steps
                    ),
                    selected_native_edges=selected_native_edges,
                    objective=selected.objective,
                    top_k=top_k,
                    exact_dense_fallback=selected.exact_dense_fallback,
                    emitted_edges=tuple(
                        CandidateEdgeTrace(
                            edge.source,
                            edge.target,
                            edge.edge_type,
                            edge.confidence,
                        )
                        for edge in selected.candidate_edges
                    ),
                    scorer_identity=_scorer_identity(request),
                    variant=self.config.variant,
                ),
            ),
        )

    def rank(self, request: ExecutionProvenanceRankingRequest) -> EpgmRetrievalResult:
        """Legacy per-node diagnostic view for path-based variants."""
        if self.config.variant == "ppr_steiner":
            raise ValueError("ppr_steiner uses rank_task so top_k is explicit")

        dense_ranked = self.dense_ranker.rank(
            TextRankingRequest(
                task_id=request.task_id,
                query_text=request.query_text,
                candidates=request.candidates,
            )
        )
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        dense_score = {node.node_id: node.score for node in dense_ranked}
        seed_relevance = _normalized_relevance(dense_ranked)
        seed_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        paths = search_epgm_paths(
            request.graph,
            seed_ids,
            seed_relevance,
            candidate_ids=frozenset(dense_score),
            config=self.config,
        )
        best_by_target = _best_path_by_target(
            paths, min_path_confidence=self.config.min_path_confidence
        )
        fused = _additive_ranking(
            dense_ranked,
            seed_relevance=seed_relevance,
            best_by_target=best_by_target,
            graph_weight=self.config.graph_weight,
        )
        ranked = tuple(
            EpgmRankedNode(
                node_id=node.node_id,
                score=node.score,
                dense_rank=dense_rank[node.node_id],
                dense_score=dense_score[node.node_id],
                graph_score=(
                    best_by_target[node.node_id].score
                    if node.node_id in best_by_target
                    else 0.0
                ),
                best_path=best_by_target.get(node.node_id),
            )
            for node in fused
        )
        return EpgmRetrievalResult(
            ranked_nodes=ranked, seed_ids=seed_ids, paths=paths
        )


def _normalized_relevance(dense_ranked: list[RankedNode]) -> dict[str, float]:
    if not dense_ranked:
        return {}
    scores = [node.score for node in dense_ranked]
    lo = min(scores)
    span = max(scores) - lo
    if span <= 0.0:
        return {node.node_id: 1.0 for node in dense_ranked}
    return {node.node_id: (node.score - lo) / span for node in dense_ranked}


def _best_path_by_target(
    paths: tuple[EpgmPath, ...],
    *,
    min_path_confidence: float = 0.0,
) -> dict[str, EpgmPath]:
    """Strongest *gate-accepted* path per target above the confidence floor.

    Rejected paths are returned by the search so each rejection stays auditable,
    so they must be filtered here. Without this check the schema gate had no
    effect on the additive ranking: paths carrying two data-flow hand-offs were
    reported as ``incomplete_path`` and still contributed their score, which
    reordered candidates on evidence the gate had already refused. The same
    applied to ``min_path_confidence``, which only the stable-insert cascade
    enforced.
    """

    best: dict[str, EpgmPath] = {}
    for path in paths:
        if not path.gate.valid:
            continue
        if path.score <= 0.0 or path.score < min_path_confidence:
            continue
        current = best.get(path.target_id)
        if current is None or path.score > current.score:
            best[path.target_id] = path
    return best


def _additive_ranking(
    dense_ranked: list[RankedNode],
    *,
    seed_relevance: dict[str, float],
    best_by_target: dict[str, EpgmPath],
    graph_weight: float,
) -> list[RankedNode]:
    """Additive fusion: graph propagation may lift a node, never demote it."""

    dense_rank = {
        node.node_id: index for index, node in enumerate(dense_ranked, start=1)
    }
    fused = [
        RankedNode(
            node.node_id,
            seed_relevance.get(node.node_id, 0.0)
            + graph_weight
            * (
                best_by_target[node.node_id].score
                if node.node_id in best_by_target
                else 0.0
            ),
        )
        for node in dense_ranked
    ]
    fused.sort(key=lambda node: (-node.score, dense_rank[node.node_id], node.node_id))
    return fused


def _select_paths(
    paths: tuple[EpgmPath, ...],
    *,
    dense_rank: dict[str, int],
    config: EpgmRetrieverConfig,
) -> dict[EpgmPath, tuple[bool, str | None]]:
    """One accepted partner per anchor and per partner, with gate reasons."""

    outcomes: dict[EpgmPath, tuple[bool, str | None]] = {}
    eligible_by_anchor: dict[str, list[EpgmPath]] = defaultdict(list)
    for path in paths:
        anchor_rank = dense_rank[path.seed_id]
        partner_rank = dense_rank[path.target_id]
        if not path.gate.valid:
            outcomes[path] = (False, path.gate.rejection_reason or "invalid_path")
        elif path.score <= 0.0:
            # A zero-relevance seed carries no evidence; emitting its edge
            # would only add noise to edge precision.
            outcomes[path] = (False, "no_path_evidence")
        elif path.score < config.min_path_confidence:
            outcomes[path] = (False, "below_path_confidence")
        elif config.fusion == "stable_insert" and (
            partner_rank <= config.preserve_dense_top_n
        ):
            outcomes[path] = (False, "protected_partner")
        elif config.fusion == "stable_insert" and partner_rank <= anchor_rank:
            outcomes[path] = (False, "partner_not_after_anchor")
        else:
            eligible_by_anchor[path.seed_id].append(path)

    winners: list[EpgmPath] = []
    for candidates in eligible_by_anchor.values():
        ordered = sorted(
            candidates,
            key=lambda path: (
                -path.score,
                dense_rank[path.target_id],
                path.target_id,
                path.node_ids,
            ),
        )
        winners.append(ordered[0])
        for path in ordered[1:]:
            outcomes[path] = (False, "lower_confidence_for_anchor")

    by_partner: dict[str, list[EpgmPath]] = defaultdict(list)
    for path in winners:
        by_partner[path.target_id].append(path)
    for candidates in by_partner.values():
        ordered = sorted(
            candidates,
            key=lambda path: (
                -path.score,
                dense_rank[path.seed_id],
                dense_rank[path.target_id],
                path.target_id,
            ),
        )
        outcomes[ordered[0]] = (True, None)
        for path in ordered[1:]:
            outcomes[path] = (False, "partner_conflict")
    return outcomes


def _stable_insert_ranking(
    dense_ranked: list[RankedNode],
    *,
    outcomes: dict[EpgmPath, tuple[bool, str | None]],
    dense_rank: dict[str, int],
    preserve_dense_top_n: int,
) -> tuple[list[RankedNode], tuple[EpgmPath, ...]]:
    """Reorder node ids only; the dense score multiset is preserved exactly."""

    final_ids = [node.node_id for node in dense_ranked]
    promoted: list[EpgmPath] = []
    accepted = sorted(
        (path for path, outcome in outcomes.items() if outcome[0]),
        key=lambda path: (
            dense_rank[path.seed_id],
            dense_rank[path.target_id],
            path.target_id,
        ),
    )
    for path in accepted:
        anchor_position = final_ids.index(path.seed_id)
        partner_position = final_ids.index(path.target_id)
        insertion_position = max(preserve_dense_top_n, anchor_position + 1)
        if partner_position <= insertion_position:
            outcomes[path] = (False, "no_effective_insertion")
            continue
        final_ids.pop(partner_position)
        final_ids.insert(insertion_position, path.target_id)
        promoted.append(path)
    if not promoted:
        return dense_ranked, ()
    score_slots = [node.score for node in dense_ranked]
    ranked = [
        RankedNode(node_id, score_slots[index])
        for index, node_id in enumerate(final_ids)
    ]
    return ranked, tuple(promoted)


def _logical_edge_type(path: EpgmPath) -> str:
    """Semantic label for the collapsed path, reported in the native trace.

    The contract-level edge in ``retrieved_subgraph`` is always ``feeds`` (the
    only provenance dependency type in ``ALLOWED_EDGE_TYPES``); this finer
    label is trace-only.
    """

    if len(path.steps) == 1:
        return path.steps[0].edge_type
    if any(
        step.edge_type == ProvenanceEdgeType.FEEDS.value for step in path.steps
    ):
        return ProvenanceEdgeType.FEEDS.value
    return ProvenanceEdgeType.DEPENDS_ON.value


def _logical_endpoints(path: EpgmPath) -> tuple[str, str]:
    """Orient the collapsed edge along the stored graph direction.

    A bidirectional walk may reach its target by going against the recorded
    orientation. Emitting ``seed -> target`` would then invert a dependency,
    so a mostly-reverse path is flipped before it becomes a public edge.
    """

    reverse_steps = sum(1 for step in path.steps if step.direction == "reverse")
    if reverse_steps * 2 > len(path.steps):
        return path.target_id, path.seed_id
    return path.seed_id, path.target_id


def _logical_edges(paths: tuple[EpgmPath, ...]) -> list[GraphEdge]:
    """Collapse each accepted path into one directed candidate-level edge."""

    edges: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        source, target = _logical_endpoints(path)
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            GraphEdge(
                source=source,
                target=target,
                edge_type="feeds",
                weight=path.score,
                directed=True,
            )
        )
    return edges


def _selected_native_edge_traces(
    transitions: tuple[TypedTransition, ...],
    request: ExecutionProvenanceRankingRequest,
) -> tuple[ProvenanceEdgeTrace, ...]:
    edge_by_key = {
        (edge.source, edge.target, edge.edge_type.value): edge
        for edge in request.graph.edges
    }
    keys = {
        (
            transition.source
            if transition.direction == "forward"
            else transition.target,
            transition.target
            if transition.direction == "forward"
            else transition.source,
            transition.edge_type,
        )
        for transition in transitions
    }
    missing = keys - edge_by_key.keys()
    if missing:
        raise ValueError(
            "selected transition does not map to stored provenance edges: "
            f"missing={sorted(missing)}"
        )
    return tuple(_edge_trace(edge_by_key[key]) for key in sorted(keys))


def _edge_traces(
    paths: tuple[EpgmPath, ...],
    request: ExecutionProvenanceRankingRequest,
) -> tuple[ProvenanceEdgeTrace, ...]:
    edge_by_key = {
        (edge.source, edge.target, edge.edge_type.value): edge
        for edge in request.graph.edges
    }
    traversed = {
        (step.source, step.target, step.edge_type)
        for path in paths
        for step in path.steps
    }
    return tuple(
        _edge_trace(edge_by_key[key]) for key in sorted(traversed) if key in edge_by_key
    )


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


__all__ = ["EpgmRankedNode", "EpgmRetrievalResult", "EpgmRetriever"]
