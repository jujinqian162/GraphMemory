from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor

from graph_memory.contracts.common import TaskId, TrainPairSampleType
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.embeddings import DenseTaskEncodingRequest
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig
from graph_memory.models.graph_retriever.contracts import (
    TextEmbeddingProvider,
    build_task_feature_groups,
)
from graph_memory.models.graph_retriever.internals.contracts import GraphBatch, LearnedEdgeBatch, TrainingBatch
from graph_memory.models.graph_retriever.internals.features import NodeFeatureBuilder
from graph_memory.models.graph_retriever.internals.tensorization import (
    ArtifactEdgeWeightPolicy,
    EdgeTensorizer,
    MessageEdgeTensors,
    UniformEdgeWeightPolicy,
)
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.validation import (
    validate_graph_batch,
    validate_graphs,
    validate_task_id_alignment,
    validate_training_batch,
)


@dataclass(frozen=True)
class TaskBatchInputs:
    """
    Already-joined artifacts needed to tensorize one task graph.
    张量化单个 task graph 所需的已 join artifact。
    """

    text_request: TextRankingRequest
    graph: MemoryGraph
    pairs: list[TrainPairRecord]
    label: EvidenceLabel | None = None


def build_edge_tensorizer(model_config: RgcnModelConfig) -> EdgeTensorizer: #TAG: Distribute
    """
    Build the edge tensorizer selected by model config.
    根据 model config 构造 edge tensorizer。
    """

    if model_config.edge_weight_policy == "uniform":
        edge_weight_policy = UniformEdgeWeightPolicy()
    elif model_config.edge_weight_policy == "artifact":
        edge_weight_policy = ArtifactEdgeWeightPolicy()
    else:
        raise ValueError(f"Unsupported edge_weight_policy: {model_config.edge_weight_policy}")
    return EdgeTensorizer(
        relation_vocab=model_config.relation_vocab,
        enabled_edge_types=frozenset(model_config.enabled_edge_types),
        edge_weight_policy=edge_weight_policy,
    )


def build_training_batches(
    *,
    ranking_requests: list[TextRankingRequest],
    graphs: list[MemoryGraph],
    pairs: list[TrainPairRecord],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    batch_size: int,
    labels: list[EvidenceLabel] | None = None,
) -> list[TrainingBatch]:
    """
    Build supervised TrainingBatch objects grouped by task graph.
    按 task graph 分组构造监督 TrainingBatch 对象。
    """

    requests_by_task_id = {request.task_id: request for request in ranking_requests}
    validate_graphs(graphs, ranking_requests)
    validate_task_id_alignment("training batch graphs", set(requests_by_task_id), {graph["task_id"] for graph in graphs})
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    labels_by_task_id = {label.task_id: label for label in labels} if labels is not None else {}
    pairs_by_task_id: dict[TaskId, list[TrainPairRecord]] = defaultdict(list)
    for pair in pairs:
        if pair["node_id"] == "q":
            raise ValueError("Training pairs must not contain node_id=q.")
        pairs_by_task_id[pair["task_id"]].append(pair)

    task_batches = [
        TaskBatchInputs(
            text_request=request,
            graph=graphs_by_task_id[request.task_id],
            pairs=pairs_by_task_id[request.task_id],
            label=labels_by_task_id.get(request.task_id),
        )
        for request in ranking_requests
        if pairs_by_task_id[request.task_id]
    ]
    return [
        _build_batch(
            tasks=task_batches[start : start + batch_size],
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            include_all_memory_nodes=False,
        )
        for start in range(0, len(task_batches), batch_size)
    ]


def build_full_ranking_batches(
    *,
    ranking_requests: list[TextRankingRequest],
    graphs: list[MemoryGraph],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    batch_size: int,
    labels: list[EvidenceLabel] | None = None,
) -> list[TrainingBatch]:
    """
    Build full-memory-node scoring batches for dev evaluation or retrieval.
    为 dev evaluation 或 retrieval 构造覆盖所有 memory node 的 scoring batch。
    """

    requests_by_task_id = {request.task_id: request for request in ranking_requests}
    validate_graphs(graphs, ranking_requests)
    validate_task_id_alignment("full ranking graphs", set(requests_by_task_id), {graph["task_id"] for graph in graphs})
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    labels_by_task_id = {label.task_id: label for label in labels} if labels is not None else {}
    task_batches = [
        TaskBatchInputs(
            text_request=request,
            graph=graphs_by_task_id[request.task_id],
            pairs=[],
            label=labels_by_task_id.get(request.task_id),
        )
        for request in ranking_requests
    ]
    return [
        _build_batch(
            tasks=task_batches[start : start + batch_size],
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            include_all_memory_nodes=True,
        )
        for start in range(0, len(task_batches), batch_size)
    ]


def move_training_batch(batch: TrainingBatch, device: torch.device | str) -> TrainingBatch:
    """
    Move tensor fields in a TrainingBatch to a device while preserving metadata.
    将 TrainingBatch 中的 tensor 字段移动到指定 device，同时保留 metadata。
    """

    graph_batch = batch.graph_batch
    moved_graph = GraphBatch(
        node_embeddings=graph_batch.node_embeddings.to(device),
        node_features=graph_batch.node_features.to(device),
        edge_index=graph_batch.edge_index.to(device),
        relation_ids=graph_batch.relation_ids.to(device),
        edge_weights=graph_batch.edge_weights.to(device),
        query_node_indices=graph_batch.query_node_indices.to(device),
        task_node_offsets=graph_batch.task_node_offsets,
        task_ids=graph_batch.task_ids,
        node_ids_by_task=graph_batch.node_ids_by_task,
    )
    moved_learned_edges = (
        LearnedEdgeBatch(
            edge_features=batch.learned_edges.edge_features.to(device),
            edge_label_mask=batch.learned_edges.edge_label_mask.to(device),
            edge_labels=batch.learned_edges.edge_labels.to(device),
        )
        if batch.learned_edges is not None
        else None
    )
    return TrainingBatch(
        graph_batch=moved_graph,
        sample_node_indices=batch.sample_node_indices.to(device),
        sample_query_indices=batch.sample_query_indices.to(device),
        sample_node_features=batch.sample_node_features.to(device),
        labels=batch.labels.to(device),
        sample_task_ids=batch.sample_task_ids,
        sample_node_ids=batch.sample_node_ids,
        sample_types=batch.sample_types,
        learned_edges=moved_learned_edges,
    )


def _build_batch(
    *,
    tasks: list[TaskBatchInputs],
    model_config: RgcnModelConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    include_all_memory_nodes: bool,
) -> TrainingBatch:
    edge_tensorizer = build_edge_tensorizer(model_config)
    feature_builder = NodeFeatureBuilder(model_config.feature_config)
    node_embeddings: list[Tensor] = []
    node_features: list[Tensor] = []
    edge_indices: list[Tensor] = []
    relation_ids: list[Tensor] = []
    edge_weights: list[Tensor] = []
    query_node_indices: list[int] = []
    task_node_offsets = [0]
    task_ids: list[str] = []
    node_ids_by_task: list[list[str]] = []
    sample_node_indices: list[int] = []
    sample_query_indices: list[int] = []
    sample_node_features: list[Tensor] = []
    labels: list[float] = []
    sample_task_ids: list[str] = []
    sample_node_ids: list[str] = []
    sample_types: list[TrainPairSampleType] = []
    learned_edge_features: list[Tensor] = []
    learned_edge_label_masks: list[Tensor] = []
    learned_edge_labels: list[Tensor] = []
    include_learned_edges = model_config.method_name == RetrievalMethodId.LEARNED_GRAPH_RGCN_RETRIEVER.value

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

    node_offset = 0
    for task, dense_features in zip(tasks, dense_features_by_task, strict=True):
        text_request = task.text_request
        task_id = text_request.task_id
        node_ids = [str(node["id"]) for node in task.graph["nodes"]]
        local_index_by_node_id = {node_id: index for index, node_id in enumerate(node_ids)}
        if "q" not in local_index_by_node_id:
            raise ValueError(f"Graph task_id={task_id} is missing q node.")

        embeddings = dense_features.node_embeddings
        features = feature_builder.build_node_features(
            node_ids=node_ids,
            seed_signals=dense_features.seed_signals,
        )
        message_edges = edge_tensorizer.tensorize_edges(task.graph)
        if message_edges.edge_index.numel() > 0:
            edge_indices.append(message_edges.edge_index + node_offset)
            relation_ids.append(message_edges.relation_ids)
            edge_weights.append(message_edges.edge_weights)
        if include_learned_edges:
            edge_batch = _build_learned_edge_batch_for_task(
                message_edges=message_edges,
                node_ids=node_ids,
                node_features=features.node_features,
                node_feature_names=features.node_feature_names,
                label=task.label,
            )
            learned_edge_features.append(edge_batch.edge_features)
            learned_edge_label_masks.append(edge_batch.edge_label_mask)
            learned_edge_labels.append(edge_batch.edge_labels)

        node_embeddings.append(embeddings)
        node_features.append(features.node_features)
        query_index = node_offset + local_index_by_node_id["q"]
        query_node_indices.append(query_index)
        task_ids.append(task_id)
        node_ids_by_task.append(node_ids)

        rows: list[TrainPairRecord]
        if include_all_memory_nodes:
            gold_nodes = set(task.label.gold_evidence_item_ids) if task.label is not None else set()
            rows = []
            for candidate in text_request.candidates:
                if candidate.item_id in gold_nodes:
                    rows.append(
                        {
                            "task_id": task_id,
                            "node_id": candidate.item_id,
                            "label": 1,
                            "sample_type": "positive",
                        }
                    )
                else:
                    rows.append(
                        {
                            "task_id": task_id,
                            "node_id": candidate.item_id,
                            "label": 0,
                            "sample_type": "easy_random",
                        }
                    )
        else:
            rows = task.pairs

        for row in rows:
            local_node_index = local_index_by_node_id[row["node_id"]]
            sample_node_indices.append(node_offset + local_node_index)
            sample_query_indices.append(query_index)
            sample_node_features.append(features.scorer_features[local_node_index])
            labels.append(float(row["label"]))
            sample_task_ids.append(task_id)
            sample_node_ids.append(row["node_id"])
            sample_types.append(row["sample_type"])

        node_offset += len(node_ids)
        task_node_offsets.append(node_offset)

    graph_batch = GraphBatch(
        node_embeddings=torch.cat(node_embeddings, dim=0),
        node_features=torch.cat(node_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long),
        relation_ids=torch.cat(relation_ids, dim=0) if relation_ids else torch.empty((0,), dtype=torch.long),
        edge_weights=torch.cat(edge_weights, dim=0) if edge_weights else torch.empty((0,), dtype=torch.float32),
        query_node_indices=torch.tensor(query_node_indices, dtype=torch.long),
        task_node_offsets=task_node_offsets,
        task_ids=task_ids,
        node_ids_by_task=node_ids_by_task,
    )
    scorer_feature_dim = len(model_config.feature_config.scorer_feature_names)
    validate_graph_batch(graph_batch)
    training_batch = TrainingBatch(
        graph_batch=graph_batch,
        sample_node_indices=torch.tensor(sample_node_indices, dtype=torch.long),
        sample_query_indices=torch.tensor(sample_query_indices, dtype=torch.long),
        sample_node_features=(
            torch.stack(sample_node_features)
            if sample_node_features
            else torch.empty((0, scorer_feature_dim), dtype=torch.float32)
        ),
        labels=torch.tensor(labels, dtype=torch.float32),
        sample_task_ids=sample_task_ids,
        sample_node_ids=sample_node_ids,
        sample_types=sample_types,
        learned_edges=(
            LearnedEdgeBatch(
                edge_features=torch.cat(learned_edge_features, dim=0)
                if learned_edge_features
                else torch.empty((0, 8), dtype=torch.float32),
                edge_label_mask=torch.cat(learned_edge_label_masks, dim=0)
                if learned_edge_label_masks
                else torch.empty((0,), dtype=torch.bool),
                edge_labels=torch.cat(learned_edge_labels, dim=0)
                if learned_edge_labels
                else torch.empty((0,), dtype=torch.float32),
            )
            if include_learned_edges
            else None
        ),
    )
    validate_training_batch(training_batch)
    return training_batch


def _build_learned_edge_batch_for_task(
    *,
    message_edges: MessageEdgeTensors,
    node_ids: list[str],
    node_features: Tensor,
    node_feature_names: tuple[str, ...],
    label: EvidenceLabel | None,
) -> LearnedEdgeBatch:
    edge_index = message_edges.edge_index
    edge_count = int(edge_index.shape[1])
    if edge_count == 0:
        return LearnedEdgeBatch(
            edge_features=torch.empty((0, 8), dtype=torch.float32),
            edge_label_mask=torch.empty((0,), dtype=torch.bool),
            edge_labels=torch.empty((0,), dtype=torch.float32),
        )

    source_indices = edge_index[0]
    target_indices = edge_index[1]
    source_features = node_features[source_indices]
    target_features = node_features[target_indices]
    seed_score_index = _node_feature_index(node_feature_names, "seed_score")
    seed_rank_index = _node_feature_index(node_feature_names, "seed_rank_percentile")
    question_flag_index = _node_feature_index(node_feature_names, "is_question_node")
    edge_features = torch.stack(
        [
            message_edges.edge_weights,
            message_edges.relation_ids.to(dtype=torch.float32),
            source_features[:, seed_score_index],
            source_features[:, seed_rank_index],
            target_features[:, seed_score_index],
            target_features[:, seed_rank_index],
            source_features[:, question_flag_index],
            target_features[:, question_flag_index],
        ],
        dim=1,
    ).to(dtype=torch.float32)

    gold_edges = set(label.gold_dependency_edges) if label is not None else set()
    has_edge_labels = bool(gold_edges)
    edge_labels: list[float] = []
    for source_index, target_index in zip(source_indices.tolist(), target_indices.tolist(), strict=True):
        source_id = node_ids[int(source_index)]
        target_id = node_ids[int(target_index)]
        edge_labels.append(1.0 if (source_id, target_id) in gold_edges or (target_id, source_id) in gold_edges else 0.0)
    return LearnedEdgeBatch(
        edge_features=edge_features,
        edge_label_mask=torch.full((edge_count,), has_edge_labels, dtype=torch.bool),
        edge_labels=torch.tensor(edge_labels, dtype=torch.float32),
    )


def _node_feature_index(node_feature_names: tuple[str, ...], feature_name: str) -> int:
    try:
        return node_feature_names.index(feature_name)
    except ValueError as error:
        available = ", ".join(node_feature_names)
        raise ValueError(f"Learned edge features require node feature={feature_name!r}; available: {available}") from error
