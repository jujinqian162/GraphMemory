from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from graph_memory.graphs.contracts import GraphEdge
from graph_memory.embeddings import SentenceEncoder
from graph_memory.models.provenance_rgcn.config import ProvenanceRgcnModelConfig
from graph_memory.models.provenance_rgcn.contracts import (
    LogicalProvenanceTransition,
    ProvenanceGraphTensor,
    ProvenanceModelOutput,
)
from graph_memory.models.provenance_rgcn.model import ExecutionProvenanceRGCN
from graph_memory.models.provenance_rgcn.tensorization import (
    move_provenance_tensor,
    tensorize_provenance_request,
)
from graph_memory.retrieval.contracts import (
    ExecutionProvenanceTrace,
    ProvenanceBindingTrace,
    ProvenanceEdgeTrace,
    ProvenancePathTrace,
    ProvenanceStructuredTransitionTrace,
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
)

_TransitionDecision: TypeAlias = Literal[
    "promoted",
    "already_above_source",
    "target_conflict",
    "lower_scoring_successor",
    "below_threshold",
    "outside_pool",
    "promotion_disabled",
    "stable_no_op",
]


@dataclass
class ExecutionProvenanceRgcnRetriever:
    model: ExecutionProvenanceRGCN
    encoder: SentenceEncoder
    config: ProvenanceRgcnModelConfig
    device: str | torch.device
    enable_edge_rerank: bool = True
    name: str = "execution_provenance_rgcn_retriever"

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        if not isinstance(request, ExecutionProvenanceRankingRequest):
            raise TypeError(
                f"{self.name} requires ExecutionProvenanceRankingRequest, "
                f"got {type(request).__name__}."
            )
        tensor = move_provenance_tensor(
            tensorize_provenance_request(
                request, encoder=self.encoder, config=self.config
            ),
            self.device,
        )
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
        return rank_provenance_output(
            tensor,
            output,
            config=self.config,
            top_k=top_k,
            enable_edge_rerank=self.enable_edge_rerank,
        )


def rank_provenance_output(
    tensor: ProvenanceGraphTensor,
    output: ProvenanceModelOutput,
    *,
    config: ProvenanceRgcnModelConfig,
    top_k: int,
    enable_edge_rerank: bool = True,
) -> RetrievalMethodResult:
    """Reconstruct one public result from a one-task tensor/output pair."""

    if tensor.task_count != 1:
        raise ValueError("rank_provenance_output requires a one-task tensor.")
    return rank_provenance_task_output(
        candidate_ids=tensor.candidate_ids_by_task[0],
        transitions=tensor.logical_transitions_by_task[0],
        output=output,
        config=config,
        top_k=top_k,
        enable_edge_rerank=enable_edge_rerank,
    )


def rank_provenance_task_output(
    *,
    candidate_ids: tuple[str, ...],
    transitions: tuple[LogicalProvenanceTransition, ...],
    output: ProvenanceModelOutput,
    config: ProvenanceRgcnModelConfig,
    top_k: int,
    enable_edge_rerank: bool = True,
) -> RetrievalMethodResult:
    """Reconstruct one task after splitting a batched model output."""

    if len(candidate_ids) != len(output.candidate_logits):
        raise ValueError("Candidate IDs and logits must have equal lengths.")
    ranked_nodes = [
        RankedNode(node_id=candidate_id, score=float(logit.detach().cpu()))
        for candidate_id, logit in zip(
            candidate_ids, output.candidate_logits, strict=True
        )
    ]
    ranked_nodes.sort(key=lambda item: (-item.score, item.node_id))
    structured = _apply_structured_reranking(
        ranked_nodes,
        transitions,
        output.edge_logits,
        config=config,
        promotion_enabled=enable_edge_rerank,
    )
    ranked_nodes = structured.ranked_nodes
    top_ids = {item.node_id for item in ranked_nodes[:top_k]}
    selected_transitions = tuple(
        (transition, score)
        for transition, score in structured.selected_transitions
        if transition.source_id in top_ids and transition.target_id in top_ids
    )
    native_edges = _native_edges(selected_transitions)
    logical_edges = _logical_edges(selected_transitions)
    return RetrievalMethodResult(
        ranked_nodes=tuple(ranked_nodes),
        trace=RetrievalTrace(
            retrieved_edges=tuple(logical_edges),
            native_trace=ExecutionProvenanceTrace(
                node_ids=tuple(
                    sorted(
                        {
                            node_id
                            for transition, _score in selected_transitions
                            for node_id in _native_node_ids(transition)
                        }
                    )
                ),
                paths=tuple(
                    ProvenancePathTrace(
                        node_ids=_native_node_ids(transition),
                        score=score,
                        semantic_relevance=score,
                        binding_consistency=1.0,
                        provenance_completeness=1.0,
                        explicit_grounding=0.0,
                        path_length_penalty=0.0,
                        invalidation_penalty=0.0,
                    )
                    for transition, score in selected_transitions
                ),
                edges=tuple(_edge_trace(edge) for edge in native_edges),
                structured_transitions=structured.traces,
                abstained_source_ids=structured.abstained_source_ids,
                structured_promotion_enabled=enable_edge_rerank,
            ),
        ),
    )


def _native_node_ids(
    transition: LogicalProvenanceTransition,
) -> tuple[str, ...]:
    return (
        transition.source_id,
        transition.native_edges[0].target,
        transition.target_id,
    )


def _native_edges(
    transitions: tuple[tuple[LogicalProvenanceTransition, float], ...],
):
    edges = {}
    for transition, _score in transitions:
        for edge in transition.native_edges:
            edges[(edge.source, edge.target, edge.edge_type.value)] = edge
    return [edges[key] for key in sorted(edges)]


def _logical_edges(
    transitions: tuple[tuple[LogicalProvenanceTransition, float], ...],
) -> list[GraphEdge]:
    return [
        GraphEdge(
            source=transition.source_id,
            target=transition.target_id,
            edge_type="sequential",
            weight=float(torch.sigmoid(torch.tensor(score))),
            directed=True,
        )
        for transition, score in transitions
    ]


@dataclass(frozen=True)
class _StructuredRankingResult:
    ranked_nodes: list[RankedNode]
    selected_transitions: tuple[tuple[LogicalProvenanceTransition, float], ...]
    traces: tuple[ProvenanceStructuredTransitionTrace, ...]
    abstained_source_ids: tuple[str, ...]


def _apply_structured_reranking(
    ranked_nodes: list[RankedNode],
    transitions: tuple[LogicalProvenanceTransition, ...],
    edge_logits: torch.Tensor,
    *,
    config: ProvenanceRgcnModelConfig,
    promotion_enabled: bool,
) -> _StructuredRankingResult:
    if len(transitions) != len(edge_logits):
        raise ValueError("Logical transition and edge-logit counts must match.")
    original_rank = {
        item.node_id: rank for rank, item in enumerate(ranked_nodes, start=1)
    }
    pool_ids = {
        item.node_id for item in ranked_nodes[: config.structured_pool_size]
    }
    seed_ids = {
        item.node_id for item in ranked_nodes[: config.structured_seed_top_s]
    }
    by_source: dict[
        str, list[tuple[LogicalProvenanceTransition, float, float]]
    ] = {}
    trace_rows: list[
        tuple[LogicalProvenanceTransition, float, _TransitionDecision]
    ] = []
    considered_sources: set[str] = set()
    for index, transition in enumerate(transitions):
        logit = float(edge_logits[index].detach().cpu())
        probability = float(torch.sigmoid(edge_logits[index]).detach().cpu())
        if transition.source_id not in seed_ids:
            continue
        considered_sources.add(transition.source_id)
        if (
            transition.source_id not in pool_ids
            or transition.target_id not in pool_ids
        ):
            trace_rows.append((transition, probability, "outside_pool"))
            continue
        if probability < config.edge_accept_threshold:
            trace_rows.append((transition, probability, "below_threshold"))
            continue
        by_source.setdefault(transition.source_id, []).append(
            (transition, logit, probability)
        )

    best_by_source: list[tuple[LogicalProvenanceTransition, float, float]] = []
    for source, candidates in sorted(by_source.items()):
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item[2],
                original_rank[item[0].target_id],
                item[0].target_id,
            ),
        )
        best_by_source.append(ordered[0])
        trace_rows.extend(
            (transition, probability, "lower_scoring_successor")
            for transition, _logit, probability in ordered[1:]
        )

    winners_by_target: dict[
        str, tuple[LogicalProvenanceTransition, float, float]
    ] = {}
    for candidate in sorted(
        best_by_source,
        key=lambda item: (
            -item[2],
            original_rank[item[0].source_id],
            original_rank[item[0].target_id],
            item[0].target_id,
        ),
    ):
        transition, _logit, probability = candidate
        if transition.target_id in winners_by_target:
            trace_rows.append((transition, probability, "target_conflict"))
            continue
        winners_by_target[transition.target_id] = candidate

    winners = sorted(
        winners_by_target.values(),
        key=lambda item: (
            original_rank[item[0].source_id],
            original_rank[item[0].target_id],
            item[0].target_id,
        ),
    )
    final_order = [item.node_id for item in ranked_nodes]
    selected: list[tuple[LogicalProvenanceTransition, float]] = []
    for transition, logit, probability in winners:
        selected.append((transition, logit))
        if not promotion_enabled:
            decision: _TransitionDecision = "promotion_disabled"
        elif original_rank[transition.target_id] < original_rank[transition.source_id]:
            decision = "already_above_source"
        else:
            before = list(final_order)
            final_order.remove(transition.target_id)
            source_index = final_order.index(transition.source_id)
            insertion_index = max(
                source_index + 1,
                min(config.preserve_node_top_n, len(final_order)),
            )
            final_order.insert(insertion_index, transition.target_id)
            decision = "promoted" if final_order != before else "stable_no_op"
        trace_rows.append((transition, probability, decision))

    score_multiset = [item.score for item in ranked_nodes]
    final_rank = {
        node_id: rank for rank, node_id in enumerate(final_order, start=1)
    }
    final_nodes = [
        RankedNode(node_id=node_id, score=score_multiset[index])
        for index, node_id in enumerate(final_order)
    ]
    winner_sources = {transition.source_id for transition, _score in selected}
    abstained = tuple(sorted(considered_sources - winner_sources))
    traces = tuple(
        ProvenanceStructuredTransitionTrace(
            source_id=transition.source_id,
            target_id=transition.target_id,
            probability=probability,
            original_source_rank=original_rank[transition.source_id],
            original_target_rank=original_rank[transition.target_id],
            final_target_rank=final_rank[transition.target_id],
            decision=decision,
        )
        for transition, probability, decision in sorted(
            trace_rows,
            key=lambda item: (
                original_rank[item[0].source_id],
                original_rank[item[0].target_id],
                item[2],
            ),
        )
    )
    return _StructuredRankingResult(
        ranked_nodes=final_nodes,
        selected_transitions=tuple(selected),
        traces=traces,
        abstained_source_ids=abstained,
    )


def _edge_trace(edge) -> ProvenanceEdgeTrace:
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
    )


__all__ = [
    "ExecutionProvenanceRgcnRetriever",
    "rank_provenance_output",
    "rank_provenance_task_output",
]
