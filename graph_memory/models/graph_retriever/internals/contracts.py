from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType

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


__all__ = ["GraphBatch", "TrainingBatch"]
