from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from graph_memory.graphs.provenance import ExecutionProvenanceEdge
from graph_memory.models.graph_batching import GraphBatch, TaskGraphTensor


@dataclass(frozen=True)
class LogicalProvenanceTransition:
    source_id: str
    target_id: str
    source_node_index: int
    target_node_index: int
    native_edges: tuple[ExecutionProvenanceEdge, ExecutionProvenanceEdge]


@dataclass(frozen=True)
class ProvenanceTaskTensor:
    """One CPU provenance graph with task-local candidate/transition indices."""

    graph_tensor: TaskGraphTensor
    node_type_ids: Tensor
    candidate_node_indices: Tensor
    candidate_ids: tuple[str, ...]
    logical_transitions: tuple[LogicalProvenanceTransition, ...]


@dataclass(frozen=True)
class ProvenanceGraphTensor:
    """One or more provenance tasks collated into a disconnected union."""

    graph_batch: GraphBatch
    node_type_ids: Tensor
    candidate_node_indices: Tensor
    candidate_query_indices: Tensor
    candidate_offsets: list[int]
    candidate_ids_by_task: tuple[tuple[str, ...], ...]
    transition_node_indices: Tensor
    transition_query_indices: Tensor
    transition_offsets: list[int]
    logical_transitions_by_task: tuple[tuple[LogicalProvenanceTransition, ...], ...]

    @property
    def task_count(self) -> int:
        return self.graph_batch.task_count

    @property
    def query_node_index(self) -> int:
        if self.task_count != 1:
            raise ValueError("query_node_index is defined only for a one-task batch.")
        return int(self.graph_batch.query_node_indices[0])

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            candidate_id
            for task_ids in self.candidate_ids_by_task
            for candidate_id in task_ids
        )

    @property
    def logical_transitions(self) -> tuple[LogicalProvenanceTransition, ...]:
        return tuple(
            transition
            for task_transitions in self.logical_transitions_by_task
            for transition in task_transitions
        )


@dataclass(frozen=True)
class ProvenanceTrainingTask:
    """One materialized provenance task with aligned v2 loss targets."""

    tensor: ProvenanceTaskTensor
    candidate_targets: Tensor
    edge_targets: Tensor


@dataclass(frozen=True)
class ProvenanceTrainingBatch:
    """Batched provenance tensor with aligned candidate and transition targets."""

    tensor: ProvenanceGraphTensor
    candidate_targets: Tensor
    edge_targets: Tensor

    @property
    def task_count(self) -> int:
        return self.tensor.task_count


@dataclass(frozen=True)
class ProvenanceModelOutput:
    node_states: Tensor
    candidate_logits: Tensor
    edge_logits: Tensor


__all__ = [
    "LogicalProvenanceTransition",
    "ProvenanceGraphTensor",
    "ProvenanceModelOutput",
    "ProvenanceTaskTensor",
    "ProvenanceTrainingBatch",
    "ProvenanceTrainingTask",
]
