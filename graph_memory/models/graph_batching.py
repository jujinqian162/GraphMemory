from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from torch.utils.data import Dataset

from graph_memory.contracts.common import NodeId, TaskId

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

TaskTensor = TypeVar("TaskTensor")


class TaskTensorDataset(Dataset[TaskTensor], Generic[TaskTensor]):
    """Immutable map-style dataset over precomputed CPU task tensors."""

    def __init__(self, items: Sequence[TaskTensor]) -> None:
        self._items = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> TaskTensor:
        return self._items[index]


@dataclass(frozen=True)
class TaskGraphTensor:
    """CPU tensor representation of one task graph before collation."""

    node_embeddings: Tensor
    node_features: Tensor
    edge_index: Tensor
    relation_ids: Tensor
    edge_weights: Tensor
    query_node_index: int
    task_id: TaskId
    node_ids: list[NodeId]


@dataclass(frozen=True)
class GraphBatch:
    """Disconnected union of one or more task graphs."""

    node_embeddings: Tensor
    node_features: Tensor
    edge_index: Tensor
    relation_ids: Tensor
    edge_weights: Tensor
    query_node_indices: Tensor
    task_node_offsets: list[int]
    task_ids: list[TaskId]
    node_ids_by_task: list[list[NodeId]]

    @property
    def task_count(self) -> int:
        return len(self.task_ids)


def collate_task_graphs(tasks: Sequence[TaskGraphTensor]) -> GraphBatch:
    """Collate task-local graph tensors into one validated disconnected union."""

    if not tasks:
        raise ValueError("Graph collation requires at least one task graph.")
    _validate_compatible_feature_shapes(tasks)

    node_embeddings: list[Tensor] = []
    node_features: list[Tensor] = []
    edge_indices: list[Tensor] = []
    relation_ids: list[Tensor] = []
    edge_weights: list[Tensor] = []
    query_node_indices: list[int] = []
    task_node_offsets = [0]
    task_ids: list[TaskId] = []
    node_ids_by_task: list[list[NodeId]] = []
    node_offset = 0

    for task in tasks:
        node_count = _validate_task_graph(task)
        node_embeddings.append(task.node_embeddings)
        node_features.append(task.node_features)
        if task.edge_index.shape[1] > 0:
            edge_indices.append(task.edge_index + node_offset)
            relation_ids.append(task.relation_ids)
            edge_weights.append(task.edge_weights)
        query_node_indices.append(node_offset + task.query_node_index)
        task_ids.append(task.task_id)
        node_ids_by_task.append(task.node_ids)
        node_offset += node_count
        task_node_offsets.append(node_offset)

    batch = GraphBatch(
        node_embeddings=torch.cat(node_embeddings, dim=0),
        node_features=torch.cat(node_features, dim=0),
        edge_index=(
            torch.cat(edge_indices, dim=1)
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        ),
        relation_ids=(
            torch.cat(relation_ids, dim=0)
            if relation_ids
            else torch.empty((0,), dtype=torch.long)
        ),
        edge_weights=(
            torch.cat(edge_weights, dim=0)
            if edge_weights
            else torch.empty((0,), dtype=torch.float32)
        ),
        query_node_indices=torch.tensor(query_node_indices, dtype=torch.long),
        task_node_offsets=task_node_offsets,
        task_ids=task_ids,
        node_ids_by_task=node_ids_by_task,
    )
    validate_graph_batch(batch)
    return batch


def move_graph_batch(batch: GraphBatch, device: torch.device | str) -> GraphBatch:
    """Move graph tensors while preserving task identity and partition metadata."""

    target = torch.device(device)
    return GraphBatch(
        node_embeddings=batch.node_embeddings.to(target),
        node_features=batch.node_features.to(target),
        edge_index=batch.edge_index.to(target),
        relation_ids=batch.relation_ids.to(target),
        edge_weights=batch.edge_weights.to(target),
        query_node_indices=batch.query_node_indices.to(target),
        task_node_offsets=batch.task_node_offsets,
        task_ids=batch.task_ids,
        node_ids_by_task=batch.node_ids_by_task,
    )


def validate_graph_batch(batch: GraphBatch) -> None:
    """Validate offsets and prove that every message edge remains task-local."""

    if len(batch.task_node_offsets) != batch.task_count + 1:
        raise ValueError("task_node_offsets must have length task_count + 1.")
    if not batch.task_node_offsets or batch.task_node_offsets[0] != 0:
        raise ValueError("task_node_offsets must start at zero.")
    node_count = int(batch.node_embeddings.shape[0])
    if batch.task_node_offsets[-1] != node_count:
        raise ValueError("task_node_offsets must end at the total node count.")
    if any(
        left >= right
        for left, right in zip(batch.task_node_offsets, batch.task_node_offsets[1:])
    ):
        raise ValueError("Every task graph must contain at least one node.")
    if batch.query_node_indices.shape != (batch.task_count,):
        raise ValueError("query_node_indices must contain one query per task.")
    for task_index, query_index in enumerate(batch.query_node_indices.tolist()):
        lower = batch.task_node_offsets[task_index]
        upper = batch.task_node_offsets[task_index + 1]
        if not lower <= query_index < upper:
            raise ValueError("A query index crosses its task node interval.")
    if batch.edge_index.shape[1] == 0:
        return
    edge_task_ids = torch.bucketize(
        batch.edge_index,
        torch.tensor(
            batch.task_node_offsets[1:-1],
            dtype=torch.long,
            device=batch.edge_index.device,
        ),
        right=True,
    )
    if not torch.equal(edge_task_ids[0], edge_task_ids[1]):
        raise ValueError("Graph batch contains a cross-task edge.")


def _validate_compatible_feature_shapes(tasks: Sequence[TaskGraphTensor]) -> None:
    embedding_dim = int(tasks[0].node_embeddings.shape[1])
    feature_dim = int(tasks[0].node_features.shape[1])
    for task in tasks:
        if (
            task.node_embeddings.ndim != 2
            or task.node_embeddings.shape[1] != embedding_dim
        ):
            raise ValueError(
                "All task node embeddings must share one feature dimension."
            )
        if task.node_features.ndim != 2 or task.node_features.shape[1] != feature_dim:
            raise ValueError("All task node features must share one feature dimension.")


def _validate_task_graph(task: TaskGraphTensor) -> int:
    node_count = len(task.node_ids)
    if node_count <= 0:
        raise ValueError(f"Task graph task_id={task.task_id!r} has no nodes.")
    if task.node_embeddings.shape[0] != node_count:
        raise ValueError("node_embeddings and node_ids counts must match.")
    if task.node_features.shape[0] != node_count:
        raise ValueError("node_features and node_ids counts must match.")
    if task.edge_index.ndim != 2 or task.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    edge_count = int(task.edge_index.shape[1])
    if task.relation_ids.shape != (edge_count,):
        raise ValueError("relation_ids must have shape [E].")
    if task.edge_weights.shape != (edge_count,):
        raise ValueError("edge_weights must have shape [E].")
    if edge_count:
        minimum = int(task.edge_index.min().item())
        maximum = int(task.edge_index.max().item())
        if minimum < 0 or maximum >= node_count:
            raise ValueError("Task-local edge index is outside the task node range.")
    if not 0 <= task.query_node_index < node_count:
        raise ValueError("Task-local query index is outside the task node range.")
    return node_count


__all__ = [
    "GraphBatch",
    "TaskGraphTensor",
    "TaskTensorDataset",
    "collate_task_graphs",
    "move_graph_batch",
    "validate_graph_batch",
]
