from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType
from graph_memory.models.graph_batching import GraphBatch

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class TrainingBatch:
    """Supervised evidence samples over a disconnected graph batch."""

    graph_batch: GraphBatch
    sample_node_indices: Tensor
    sample_query_indices: Tensor
    sample_node_features: Tensor
    labels: Tensor
    sample_task_ids: list[TaskId]
    sample_node_ids: list[NodeId]
    sample_types: list[TrainPairSampleType]


__all__ = ["GraphBatch", "TrainingBatch"]
