from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.graphs.provenance import ExecutionProvenanceEdge, ProvenanceEdgeType
from graph_memory.retrieval.contracts import (
    CandidateEdgeTrace,
    DenseRankTrace,
    ProvenanceBindingTrace,
    ProvenanceEdgeTrace,
    StatelessExecutionProvenanceTrace,
    StatelessProvenancePathTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.methods.execution_provenance.config import (
    ExecutionProvenanceConfig,
)
from graph_memory.retrieval.methods.execution_provenance.search import (
    ProvenancePathEvaluation,
    search_provenance_paths,
)
from graph_memory.retrieval.methods.flat.dense import DenseTaskRetriever
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
    TextRankingRequest,
)


@dataclass(frozen=True)
class ExecutionProvenanceRetriever:
    dense_ranker: DenseTaskRetriever
    config: ExecutionProvenanceConfig = ExecutionProvenanceConfig()
    name: str = "execution_provenance_retriever"

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
        dense_ranked = self.dense_ranker.rank(
            TextRankingRequest(request.task_id, request.query_text, request.candidates)
        )
        dense_rank = {
            node.node_id: index for index, node in enumerate(dense_ranked, start=1)
        }
        seed_ids = tuple(
            node.node_id for node in dense_ranked[: self.config.seed_top_s]
        )
        evaluations = search_provenance_paths(
            request,
            seed_ids,
            config=self.config,
        )
        outcomes = _select_path_proposals(
            evaluations,
            dense_rank=dense_rank,
            config=self.config,
        )
        final_ids, moved = _stable_insert(
            [node.node_id for node in dense_ranked],
            outcomes=outcomes,
            dense_rank=dense_rank,
            preserve_dense_top_n=self.config.preserve_dense_top_n,
        )
        actual = {evaluation for evaluation, _old_rank, _new_rank in moved}
        for evaluation in tuple(outcomes):
            accepted, reason = outcomes[evaluation]
            if accepted and evaluation not in actual:
                outcomes[evaluation] = (False, reason or "no_effective_insertion")
        if not moved:
            ranked_nodes = dense_ranked
        else:
            score_slots = [node.score for node in dense_ranked]
            ranked_nodes = [
                RankedNode(node_id, score_slots[index])
                for index, node_id in enumerate(final_ids)
            ]
        final_rank = {
            node.node_id: index for index, node in enumerate(ranked_nodes, start=1)
        }
        top_ids = set(final_ids[:top_k])
        emitted = tuple(
            evaluation
            for evaluation, _old_rank, _new_rank in moved
            if evaluation.anchor_id in top_ids and evaluation.partner_id in top_ids
        )
        retrieved_edges = [_logical_edge(evaluation) for evaluation in emitted]
        evaluated_edges = _evaluated_edges(evaluations)
        trace = StatelessExecutionProvenanceTrace(
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
                    anchor_id=evaluation.anchor_id,
                    partner_id=evaluation.partner_id,
                    node_ids=evaluation.path.node_ids,
                    path_confidence=evaluation.score.path_confidence,
                    binding_valid=evaluation.score.binding_valid,
                    completeness_valid=evaluation.score.completeness_valid,
                    lifecycle_valid=evaluation.score.lifecycle_valid,
                    accepted=(outcomes[evaluation][0] and evaluation in actual),
                    rejection_reason=(
                        None
                        if outcomes[evaluation][0] and evaluation in actual
                        else outcomes[evaluation][1]
                    ),
                    original_partner_rank=dense_rank[evaluation.partner_id],
                    final_partner_rank=final_rank[evaluation.partner_id],
                )
                for evaluation in evaluations
            ),
            edges=tuple(_edge_trace(edge) for edge in evaluated_edges),
            protected_prefix=tuple(
                node.node_id
                for node in dense_ranked[: self.config.preserve_dense_top_n]
            ),
            exact_dense_fallback=not moved,
            emitted_edges=tuple(
                CandidateEdgeTrace(
                    source=evaluation.anchor_id,
                    target=evaluation.partner_id,
                    edge_type="feeds",
                    confidence=evaluation.score.path_confidence,
                )
                for evaluation in emitted
            ),
            scorer_identity=_semantic_scorer_identity(request),
        )
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=retrieved_edges,
                native_trace=trace,
            ),
        )


def _select_path_proposals(
    evaluations: tuple[ProvenancePathEvaluation, ...],
    *,
    dense_rank: dict[str, int],
    config: ExecutionProvenanceConfig,
) -> dict[ProvenancePathEvaluation, tuple[bool, str | None]]:
    outcomes: dict[ProvenancePathEvaluation, tuple[bool, str | None]] = {}
    eligible_by_anchor: dict[str, list[ProvenancePathEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        partner_rank = dense_rank[evaluation.partner_id]
        anchor_rank = dense_rank[evaluation.anchor_id]
        if not evaluation.score.valid:
            outcomes[evaluation] = (
                False,
                evaluation.score.rejection_reason or "invalid_path",
            )
        elif evaluation.score.path_confidence < config.min_path_confidence:
            outcomes[evaluation] = (False, "below_path_confidence")
        elif partner_rank <= config.preserve_dense_top_n:
            outcomes[evaluation] = (False, "protected_partner")
        elif partner_rank <= anchor_rank:
            outcomes[evaluation] = (False, "partner_not_after_anchor")
        else:
            eligible_by_anchor[evaluation.anchor_id].append(evaluation)
    anchor_winners: list[ProvenancePathEvaluation] = []
    for candidates in eligible_by_anchor.values():
        ordered = sorted(
            candidates,
            key=lambda evaluation: (
                -evaluation.score.path_confidence,
                dense_rank[evaluation.partner_id],
                evaluation.partner_id,
                evaluation.path.node_ids,
            ),
        )
        anchor_winners.append(ordered[0])
        for evaluation in ordered[1:]:
            outcomes[evaluation] = (False, "lower_confidence_for_anchor")
    by_partner: dict[str, list[ProvenancePathEvaluation]] = defaultdict(list)
    for evaluation in anchor_winners:
        by_partner[evaluation.partner_id].append(evaluation)
    for candidates in by_partner.values():
        ordered = sorted(
            candidates,
            key=lambda evaluation: (
                -evaluation.score.path_confidence,
                dense_rank[evaluation.anchor_id],
                dense_rank[evaluation.partner_id],
                evaluation.partner_id,
            ),
        )
        outcomes[ordered[0]] = (True, None)
        for evaluation in ordered[1:]:
            outcomes[evaluation] = (False, "partner_conflict")
    return outcomes


def _stable_insert(
    dense_ids: list[str],
    *,
    outcomes: dict[ProvenancePathEvaluation, tuple[bool, str | None]],
    dense_rank: dict[str, int],
    preserve_dense_top_n: int,
) -> tuple[list[str], list[tuple[ProvenancePathEvaluation, int, int]]]:
    final_ids = list(dense_ids)
    moved: list[tuple[ProvenancePathEvaluation, int, int]] = []
    accepted = sorted(
        (evaluation for evaluation, outcome in outcomes.items() if outcome[0]),
        key=lambda evaluation: (
            dense_rank[evaluation.anchor_id],
            dense_rank[evaluation.partner_id],
            evaluation.partner_id,
        ),
    )
    for evaluation in accepted:
        anchor_position = final_ids.index(evaluation.anchor_id)
        partner_position = final_ids.index(evaluation.partner_id)
        insertion_position = max(preserve_dense_top_n, anchor_position + 1)
        if partner_position <= insertion_position:
            outcomes[evaluation] = (False, "no_effective_insertion")
            continue
        final_ids.pop(partner_position)
        final_ids.insert(insertion_position, evaluation.partner_id)
        moved.append((evaluation, partner_position + 1, insertion_position + 1))
    return final_ids, moved


def _logical_edge(evaluation: ProvenancePathEvaluation) -> GraphEdge:
    return {
        "source": evaluation.anchor_id,
        "target": evaluation.partner_id,
        "edge_type": "feeds",
        "weight": evaluation.score.path_confidence,
        "directed": True,
    }


def _evaluated_edges(
    evaluations: tuple[ProvenancePathEvaluation, ...],
) -> tuple[ExecutionProvenanceEdge, ...]:
    by_key = {
        (edge.source, edge.target, edge.edge_type.value): edge
        for evaluation in evaluations
        for edge in evaluation.path.edges
    }
    return tuple(by_key[key] for key in sorted(by_key))


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


def _semantic_scorer_identity(request: ExecutionProvenanceRankingRequest) -> str:
    identities = {
        value
        for edge in request.graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
        if isinstance((value := edge.metadata.get("semantic_scorer")), str)
    }
    return ",".join(sorted(identities)) or "unknown"


__all__ = ["ExecutionProvenanceRetriever"]
