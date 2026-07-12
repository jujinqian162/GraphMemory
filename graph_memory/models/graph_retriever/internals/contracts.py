from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType
from graph_memory.evaluation.requests import EvidenceLabel

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class GraphBatch:
    """
    Tensorized batch of one or more task graphs for message passing.
    一个或多个 task graph 的 message passing 张量化 batch。
    """

    node_embeddings: Tensor
    node_features: Tensor
    edge_index: Tensor
    relation_ids: Tensor
    edge_weights: Tensor
    query_node_indices: Tensor
    task_node_offsets: list[int]
    task_ids: list[TaskId]
    node_ids_by_task: list[list[NodeId]]


@dataclass(frozen=True)
class TrainingBatch:
    """
    Supervised sample batch over a tensorized graph batch.
    基于张量化 graph batch 的监督样本 batch。
    """

    graph_batch: GraphBatch
    sample_node_indices: Tensor
    sample_query_indices: Tensor
    sample_node_features: Tensor
    labels: Tensor
    sample_task_ids: list[TaskId]
    sample_node_ids: list[NodeId]
    sample_types: list[TrainPairSampleType]
    candidate_node_indices: Tensor | None = None
    candidate_query_indices: Tensor | None = None
    candidate_node_features: Tensor | None = None
    candidate_task_offsets: list[int] | None = None
    candidate_node_ids_by_task: list[list[NodeId]] | None = None
    evidence_labels: list[EvidenceLabel | None] | None = None


@dataclass(frozen=True)
class EncodedGraphState:
    graph_batch: GraphBatch
    node_states: Tensor
    question_node_indices: Tensor
    task_node_offsets: list[int]
    task_ids: list[TaskId]
    node_ids_by_task: list[list[NodeId]]
    candidate_node_indices: Tensor
    candidate_query_indices: Tensor
    candidate_node_features: Tensor
    candidate_task_offsets: list[int]
    candidate_node_ids_by_task: list[list[NodeId]]
    base_node_logits: Tensor

    def candidate_states_for_task(self, task_index: int) -> Tensor:
        start = self.candidate_task_offsets[task_index]
        end = self.candidate_task_offsets[task_index + 1]
        return self.node_states[self.candidate_node_indices[start:end]]

    def base_logits_for_task(self, task_index: int) -> Tensor:
        start = self.candidate_task_offsets[task_index]
        end = self.candidate_task_offsets[task_index + 1]
        return self.base_node_logits[start:end]


__all__ = ["EncodedGraphState", "GraphBatch", "TrainingBatch"]
