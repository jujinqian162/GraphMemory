from __future__ import annotations

from dataclasses import dataclass

import torch

from graph_memory.contracts.graphs import GraphEdge
from graph_memory.embeddings import SentenceEncoder
from graph_memory.models.provenance_rgcn.config import ProvenanceRgcnModelConfig
from graph_memory.models.provenance_rgcn.contracts import LogicalProvenanceTransition
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
    RankedNode,
    RetrievalMethodResult,
    RetrievalTrace,
)
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    RankingMethodRequest,
)


@dataclass
class ExecutionProvenanceRgcnRetriever:
    model: ExecutionProvenanceRGCN
    encoder: SentenceEncoder
    config: ProvenanceRgcnModelConfig
    device: str | torch.device = "cpu"
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
        ranked_nodes = [
            RankedNode(
                candidate_id,
                float(output.candidate_logits[index].detach().cpu()),
            )
            for index, candidate_id in enumerate(tensor.candidate_ids)
        ]
        ranked_nodes.sort(key=lambda item: (-item.score, item.node_id))
        top_ids = {item.node_id for item in ranked_nodes[:top_k]}
        selected_transitions = _select_transitions(
            tensor.logical_transitions,
            output.edge_logits,
            top_ids,
        )
        native_edges = _native_edges(selected_transitions)
        logical_edges = _logical_edges(selected_transitions)
        return RetrievalMethodResult(
            ranked_nodes=ranked_nodes,
            trace=RetrievalTrace(
                retrieved_edges=logical_edges,
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
        {
            "source": transition.source_id,
            "target": transition.target_id,
            "edge_type": "sequential",
            "weight": float(torch.sigmoid(torch.tensor(score))),
            "directed": True,
        }
        for transition, score in transitions
    ]


def _select_transitions(
    transitions: tuple[LogicalProvenanceTransition, ...],
    edge_logits: torch.Tensor,
    selected_ids: set[str],
) -> tuple[tuple[LogicalProvenanceTransition, float], ...]:
    by_source: dict[str, list[tuple[LogicalProvenanceTransition, float]]] = {}
    for index, transition in enumerate(transitions):
        if (
            transition.source_id not in selected_ids
            or transition.target_id not in selected_ids
        ):
            continue
        score = float(edge_logits[index].detach().cpu())
        by_source.setdefault(transition.source_id, []).append((transition, score))
    selected = [
        sorted(
            candidates,
            key=lambda item: (-item[1], item[0].target_id),
        )[0]
        for _source, candidates in sorted(by_source.items())
    ]
    return tuple(selected)


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


__all__ = ["ExecutionProvenanceRgcnRetriever"]
