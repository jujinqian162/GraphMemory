from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from graph_memory.contracts.common import TaskId, TrainPairSampleType
from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.embeddings import DenseTaskEncodingRequest
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_batching import (
    TaskGraphTensor,
    TaskTensorDataset,
    collate_task_graphs,
    move_graph_batch,
)
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig
from graph_memory.models.graph_retriever.contracts import (
    TextEmbeddingProvider,
    build_task_feature_groups,
)
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.models.graph_retriever.internals.features import NodeFeatureBuilder
from graph_memory.models.graph_retriever.internals.tensorization import (
    ArtifactEdgeWeightPolicy,
    EdgeTensorizer,
    UniformEdgeWeightPolicy,
)
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TaskBatchInputs:
    """Already-joined artifacts needed to tensorize one evidence task graph."""

    text_request: TextRankingRequest
    graph: EvidenceGraph
    pairs: list[TrainPairRecord]
    label: EvidenceLabel | None = None


@dataclass(frozen=True)
class EvidenceTaskTensor:
    """One materialized CPU evidence task and its task-local supervision."""

    graph_tensor: TaskGraphTensor
    sample_node_indices: Tensor
    sample_node_features: Tensor
    labels: Tensor
    sample_node_ids: list[str]
    sample_types: list[TrainPairSampleType]


def build_edge_tensorizer(model_config: RgcnModelConfig) -> EdgeTensorizer:
    """Build the edge tensorizer selected by model config."""

    if model_config.edge_weight_policy == "uniform":
        edge_weight_policy = UniformEdgeWeightPolicy()
    elif model_config.edge_weight_policy == "artifact":
        edge_weight_policy = ArtifactEdgeWeightPolicy()
    else:
        raise ValueError(
            f"Unsupported edge_weight_policy: {model_config.edge_weight_policy}"
        )
    return EdgeTensorizer(
        relation_vocab=model_config.relation_vocab,
        enabled_edge_types=frozenset(model_config.enabled_edge_types),
        edge_weight_policy=edge_weight_policy,
    )


def materialize_training_tasks(
    *,
    ranking_requests: list[TextRankingRequest],
    graphs: list[EvidenceGraph],
    pairs: list[TrainPairRecord],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    progress_desc: str | None = None,
) -> list[EvidenceTaskTensor]:
    """Materialize one CPU tensor per supervised evidence task."""

    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    pairs_by_task_id: dict[TaskId, list[TrainPairRecord]] = defaultdict(list)
    for pair in pairs:
        pairs_by_task_id[pair["task_id"]].append(pair)
    inputs = [
        TaskBatchInputs(
            text_request=request,
            graph=graphs_by_task_id[request.task_id],
            pairs=pairs_by_task_id[request.task_id],
        )
        for request in ranking_requests
        if pairs_by_task_id[request.task_id]
    ]
    return _materialize_tasks(
        inputs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        include_all_memory_nodes=False,
        progress_desc=progress_desc,
    )


def materialize_full_ranking_tasks(
    *,
    ranking_requests: list[TextRankingRequest],
    graphs: list[EvidenceGraph],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    labels: list[EvidenceLabel] | None = None,
    progress_desc: str | None = None,
) -> list[EvidenceTaskTensor]:
    """Materialize ordered CPU tensors for full evidence ranking."""

    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    labels_by_task_id = (
        {label.task_id: label for label in labels} if labels is not None else {}
    )
    inputs = [
        TaskBatchInputs(
            text_request=request,
            graph=graphs_by_task_id[request.task_id],
            pairs=[],
            label=labels_by_task_id.get(request.task_id),
        )
        for request in ranking_requests
    ]
    return _materialize_tasks(
        inputs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        include_all_memory_nodes=True,
        progress_desc=progress_desc,
    )


def build_evidence_dataloader(
    tasks: Sequence[EvidenceTaskTensor],
    *,
    per_device_graph_batch_size: int,
    shuffle: bool,
    random_seed: int,
) -> DataLoader[TrainingBatch]:
    """Build a deterministic map-style DataLoader over materialized task tensors."""

    if per_device_graph_batch_size <= 0:
        raise ValueError("per_device_graph_batch_size must be positive.")
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return cast(
        DataLoader[TrainingBatch],
        DataLoader(
            TaskTensorDataset(tasks),
            batch_size=per_device_graph_batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_evidence_tasks,
        ),
    )


def collate_evidence_tasks(tasks: Sequence[EvidenceTaskTensor]) -> TrainingBatch:
    """Collate task-local evidence tensors after DataLoader sampling."""

    if not tasks:
        raise ValueError("Evidence collation requires at least one task.")
    for task in tasks:
        _validate_evidence_task(task)
    graph_batch = collate_task_graphs([task.graph_tensor for task in tasks])
    sample_node_indices: list[Tensor] = []
    sample_query_indices: list[Tensor] = []
    sample_node_features: list[Tensor] = []
    labels: list[Tensor] = []
    sample_task_ids: list[str] = []
    sample_node_ids: list[str] = []
    sample_types: list[TrainPairSampleType] = []
    for task_index, task in enumerate(tasks):
        sample_count = int(task.labels.shape[0])
        node_offset = graph_batch.task_node_offsets[task_index]
        sample_node_indices.append(task.sample_node_indices + node_offset)
        sample_query_indices.append(
            torch.full(
                (sample_count,),
                int(graph_batch.query_node_indices[task_index]),
                dtype=torch.long,
            )
        )
        sample_node_features.append(task.sample_node_features)
        labels.append(task.labels)
        sample_task_ids.extend([task.graph_tensor.task_id] * sample_count)
        sample_node_ids.extend(task.sample_node_ids)
        sample_types.extend(task.sample_types)
    feature_dim = int(tasks[0].sample_node_features.shape[1])
    return TrainingBatch(
        graph_batch=graph_batch,
        sample_node_indices=torch.cat(sample_node_indices),
        sample_query_indices=torch.cat(sample_query_indices),
        sample_node_features=(
            torch.cat(sample_node_features, dim=0)
            if sample_node_features
            else torch.empty((0, feature_dim), dtype=torch.float32)
        ),
        labels=torch.cat(labels),
        sample_task_ids=sample_task_ids,
        sample_node_ids=sample_node_ids,
        sample_types=sample_types,
    )


def move_training_batch(
    batch: TrainingBatch, device: torch.device | str
) -> TrainingBatch:
    """Move evidence tensor fields while preserving CPU metadata."""

    return TrainingBatch(
        graph_batch=move_graph_batch(batch.graph_batch, device),
        sample_node_indices=batch.sample_node_indices.to(device),
        sample_query_indices=batch.sample_query_indices.to(device),
        sample_node_features=batch.sample_node_features.to(device),
        labels=batch.labels.to(device),
        sample_task_ids=batch.sample_task_ids,
        sample_node_ids=batch.sample_node_ids,
        sample_types=batch.sample_types,
    )


def _validate_evidence_task(task: EvidenceTaskTensor) -> None:
    sample_count = int(task.labels.shape[0])
    if task.sample_node_indices.shape != (sample_count,):
        raise ValueError("Evidence sample indices and labels must have equal lengths.")
    if (
        task.sample_node_features.ndim != 2
        or task.sample_node_features.shape[0] != sample_count
    ):
        raise ValueError("Evidence sample features and labels must have equal lengths.")
    if (
        len(task.sample_node_ids) != sample_count
        or len(task.sample_types) != sample_count
    ):
        raise ValueError("Evidence sample metadata and labels must have equal lengths.")
    if sample_count:
        node_count = len(task.graph_tensor.node_ids)
        minimum = int(task.sample_node_indices.min().item())
        maximum = int(task.sample_node_indices.max().item())
        if minimum < 0 or maximum >= node_count:
            raise ValueError("An evidence sample index crosses its task node interval.")


def _materialize_tasks(
    tasks: Sequence[TaskBatchInputs],
    *,
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    include_all_memory_nodes: bool,
    progress_desc: str | None,
) -> list[EvidenceTaskTensor]:
    if not tasks:
        return []
    requests = [
        DenseTaskEncodingRequest(
            ranking_request=task.text_request,
            node_ids=tuple(str(node["id"]) for node in task.graph["nodes"]),
        )
        for task in tasks
    ]
    dense_features_by_task = build_task_feature_groups(
        text_embedding_provider,
        seed_signal_provider,
        requests,
    )
    builder = partial(
        _materialize_task,
        model_config=model_config,
        include_all_memory_nodes=include_all_memory_nodes,
    )
    task_features = zip(tasks, dense_features_by_task, strict=True)
    if progress_desc is not None:
        task_features = tqdm(
            task_features,
            total=len(tasks),
            desc=progress_desc,
            unit="task",
        )
    return [
        builder(task=task, dense_features=dense_features)
        for task, dense_features in task_features
    ]


def _materialize_task(
    *,
    task: TaskBatchInputs,
    dense_features,
    model_config: RgcnModelConfig,
    include_all_memory_nodes: bool,
) -> EvidenceTaskTensor:
    edge_tensorizer = build_edge_tensorizer(model_config)
    feature_builder = NodeFeatureBuilder(model_config.feature_config)
    text_request = task.text_request
    task_id = text_request.task_id
    node_ids = [str(node["id"]) for node in task.graph["nodes"]]
    local_index_by_node_id = {node_id: index for index, node_id in enumerate(node_ids)}
    if "q" not in local_index_by_node_id:
        raise ValueError(f"Graph task_id={task_id} is missing q node.")

    features = feature_builder.build_node_features(
        node_ids=node_ids,
        seed_signals=dense_features.seed_signals,
    )
    message_edges = edge_tensorizer.tensorize_edges(task.graph)
    graph_tensor = TaskGraphTensor(
        node_embeddings=dense_features.node_embeddings,
        node_features=features.node_features,
        edge_index=message_edges.edge_index,
        relation_ids=message_edges.relation_ids,
        edge_weights=message_edges.edge_weights,
        query_node_index=local_index_by_node_id["q"],
        task_id=task_id,
        node_ids=node_ids,
    )

    rows: list[TrainPairRecord]
    if include_all_memory_nodes:
        gold_nodes = (
            set(task.label.gold_evidence_item_ids) if task.label is not None else set()
        )
        rows = []
        for candidate in text_request.candidates:
            is_gold = candidate.item_id in gold_nodes
            sample_type: TrainPairSampleType = "positive" if is_gold else "easy_random"
            row_label: Literal[0, 1] = 1 if is_gold else 0
            row: TrainPairRecord = {
                "task_id": task_id,
                "node_id": candidate.item_id,
                "label": row_label,
                "sample_type": sample_type,
            }
            rows.append(row)
    else:
        rows = task.pairs

    sample_node_indices = [local_index_by_node_id[row["node_id"]] for row in rows]
    return EvidenceTaskTensor(
        graph_tensor=graph_tensor,
        sample_node_indices=torch.tensor(sample_node_indices, dtype=torch.long),
        sample_node_features=(
            features.scorer_features[sample_node_indices]
            if sample_node_indices
            else torch.empty(
                (0, len(model_config.feature_config.scorer_feature_names)),
                dtype=torch.float32,
            )
        ),
        labels=torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32),
        sample_node_ids=[row["node_id"] for row in rows],
        sample_types=[row["sample_type"] for row in rows],
    )


__all__ = [
    "EvidenceTaskTensor",
    "TaskBatchInputs",
    "build_edge_tensorizer",
    "build_evidence_dataloader",
    "collate_evidence_tasks",
    "materialize_full_ranking_tasks",
    "materialize_training_tasks",
    "move_training_batch",
]
