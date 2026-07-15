from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from graph_memory.graphs.provenance import ExecutionProvenanceEdge
from graph_memory.models.graph_retriever.internals.contracts import GraphBatch


@dataclass(frozen=True)
class LogicalProvenanceTransition:
    source_id: str
    target_id: str
    source_node_index: int
    target_node_index: int
    native_edges: tuple[ExecutionProvenanceEdge, ExecutionProvenanceEdge]


@dataclass(frozen=True)
class ProvenanceGraphTensor:
    graph_batch: GraphBatch
    node_type_ids: Tensor
    query_node_index: int
    candidate_node_indices: Tensor
    candidate_ids: tuple[str, ...]
    logical_transitions: tuple[LogicalProvenanceTransition, ...]


@dataclass(frozen=True)
class ProvenanceModelOutput:
    node_states: Tensor
    candidate_logits: Tensor
    edge_logits: Tensor


__all__ = [
    "LogicalProvenanceTransition",
    "ProvenanceGraphTensor",
    "ProvenanceModelOutput",
]
