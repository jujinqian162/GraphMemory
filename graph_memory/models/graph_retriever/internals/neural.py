from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor, nn

from dataclasses import dataclass

from graph_memory.models.graph_retriever.internals.contracts import (
    EncodedGraphState,
    GraphBatch,
    TrainingBatch,
)


class GraphEncoder(Protocol):
    """
    Replaceable graph encoder over batch-flattened node states.
    基于 batch-flattened node state 的可替换图编码器。

    Methods / 方法:
    - forward: Return encoded node states with the same first dimension as input node states.
      forward：返回编码后的 node states，第一维必须与输入 node states 一致。
    """

    def forward(self, batch: GraphBatch, node_states: Tensor) -> Tensor: ...

    def __call__(self, batch: GraphBatch, node_states: Tensor) -> Tensor: ...


class IdentityGraphEncoder(nn.Module):
    """
    Graph encoder that leaves node states unchanged.
    不改变 node state 的图编码器。

    Methods / 方法:
    - forward: Return the exact input node state tensor.
      forward：返回原始输入 node state tensor。
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: GraphBatch, node_states: Tensor) -> Tensor:
        return node_states


class MessageTransform(Protocol):
    """
    Replaceable relation-aware message transform.
    可替换的 relation-aware message transform。

    Methods / 方法:
    - forward: Transform source node states for message edges using relation ids.
      forward：根据 relation id 转换 message edge 的 source node state。
    """

    def forward(self, source_states: Tensor, relation_ids: Tensor) -> Tensor: ...


MessageTransformFactory = Callable[[], MessageTransform]


class TypedRelationTransform(nn.Module):
    """
    Relation transform with one linear map per relation id.
    每个 relation id 使用一个线性变换的 relation transform。

    Fields / 字段:
    - relation_linears: Ordered linear transforms indexed by relation id.
      relation_linears：按 relation id 索引的有序线性变换。
    """

    def __init__(self, *, hidden_dim: int, num_relations: int) -> None:
        super().__init__()
        self.relation_linears = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)
        )

    def forward(self, source_states: Tensor, relation_ids: Tensor) -> Tensor:
        if relation_ids.ndim != 1:
            raise ValueError("relation_ids must be a 1D tensor.")
        if relation_ids.shape[0] != source_states.shape[0]:
            raise ValueError("relation_ids must have the same length as source_states.")
        if relation_ids.numel() > 0:
            min_relation_id = int(relation_ids.min().item())
            max_relation_id = int(relation_ids.max().item())
            if min_relation_id < 0 or max_relation_id >= len(self.relation_linears):
                raise ValueError(
                    f"relation_ids must be in [0, {len(self.relation_linears)}), "
                    f"got min={min_relation_id} max={max_relation_id}."
                )

        transformed = torch.empty_like(source_states)
        for relation_id, linear in enumerate(self.relation_linears):
            mask = relation_ids == relation_id
            if bool(mask.any()):
                transformed[mask] = linear(source_states[mask])
        return transformed


class SharedRelationTransform(nn.Module):
    """
    Relation transform that shares one linear map across all relation ids.
    所有 relation id 共用一个线性变换的 relation transform。

    Fields / 字段:
    - message_linear: Shared linear transform for all message edges.
      message_linear：所有 message edge 共用的线性变换。
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.message_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, source_states: Tensor, relation_ids: Tensor) -> Tensor:
        return self.message_linear(source_states)


class RelationalGraphConvLayer(nn.Module):
    """
    One R-GCN message passing layer over batch-flattened graph tensors.
    基于 batch-flattened graph tensor 的单层 R-GCN message passing。

    Fields / 字段:
    - message_transform: Relation transform applied to source node states.
      message_transform：应用在 source node state 上的 relation transform。
    - self_linear: Linear self-state transform.
      self_linear：节点自身状态的线性变换。
    - dropout: Dropout applied to aggregated messages.
      dropout：应用在聚合 message 上的 dropout。
    - layer_norm: Layer normalization after residual-style message combination.
      layer_norm：message 组合后的 layer normalization。
    """

    def __init__(
        self, *, hidden_dim: int, message_transform: MessageTransform, dropout: float
    ) -> None:
        super().__init__()
        self.message_transform = message_transform
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: GraphBatch, node_states: Tensor) -> Tensor:
        if batch.edge_index.numel() == 0:
            return self.layer_norm(torch.relu(self.self_linear(node_states)))

        source_indices = batch.edge_index[0]
        target_indices = batch.edge_index[1]
        source_states = node_states[source_indices]
        messages = self.message_transform.forward(source_states, batch.relation_ids)
        messages = messages * batch.edge_weights.unsqueeze(1)
        messages = messages * _relation_degree_norm(
            target_indices=target_indices,
            relation_ids=batch.relation_ids,
            num_nodes=node_states.shape[0],
        ).unsqueeze(1)

        aggregated = torch.zeros_like(node_states)
        aggregated.index_add_(0, target_indices, messages)
        next_states = self.self_linear(node_states) + self.dropout(aggregated)
        return self.layer_norm(torch.relu(next_states))


class RGCNGraphEncoder(nn.Module):
    """
    Stacked R-GCN graph encoder.
    堆叠式 R-GCN 图编码器。

    Fields / 字段:
    - layers: Ordered R-GCN layers applied to node states.
      layers：按顺序应用到 node state 的 R-GCN 层。
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_relations: int,
        num_layers: int,
        message_transform_factory: MessageTransformFactory,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if num_layers == 0:
            self.layers = nn.ModuleList()
            return

        layers: list[RelationalGraphConvLayer] = [
            RelationalGraphConvLayer(
                hidden_dim=hidden_dim,
                message_transform=message_transform_factory(),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, batch: GraphBatch, node_states: Tensor) -> Tensor:
        encoded = node_states
        for layer in self.layers:
            encoded = layer(batch, encoded)
        return encoded


class EvidenceNodeScorer(nn.Module):
    """
    MLP scorer that maps memory/query node states to one evidence logit per sample.
    将 memory/query node state 映射为每个 sample 一个 evidence logit 的 MLP scorer。

    Fields / 字段:
    - network: MLP over `[h_node, h_query, h_node * h_query, sample_node_features]`.
      network：作用在 `[h_node, h_query, h_node * h_query, sample_node_features]` 上的 MLP。
    """

    def __init__(
        self, *, hidden_dim: int, scorer_feature_dim: int, dropout: float
    ) -> None:
        super().__init__()
        input_dim = hidden_dim * 3 + scorer_feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, *, node_states: Tensor, query_states: Tensor, sample_node_features: Tensor
    ) -> Tensor:
        scorer_input = torch.cat(
            [
                node_states,
                query_states,
                node_states * query_states,
                sample_node_features,
            ],
            dim=1,
        )
        return self.network(scorer_input).squeeze(-1)


@dataclass(frozen=True)
class DecoderActionScores:
    candidate_logits: Tensor
    stop_logit: Tensor


class EvidenceScoringModel(nn.Module):
    """
    Trainable evidence node scorer with frozen text embeddings and replaceable graph encoder.
    使用冻结文本 embedding 和可替换图编码器的可训练 evidence node scorer。

    Fields / 字段:
    - input_projection: Projects text embeddings plus numeric node features to hidden states.
      input_projection：将 text embedding 和数值 node feature 投影到 hidden state。
    - graph_encoder: Replaceable graph encoder such as identity or R-GCN.
      graph_encoder：可替换图编码器，例如 identity 或 R-GCN。
    - scorer: Evidence node scorer that returns one logit per supervised sample.
      scorer：为每个监督 sample 返回一个 logit 的 evidence node scorer。
    """

    def __init__(
        self,
        *,
        encoder_dim: int,
        node_feature_dim: int,
        hidden_dim: int,
        graph_encoder: GraphEncoder,
        scorer_feature_dim: int,
        dropout: float,
        decoder_hidden_dim: int | None = None,
        step_embedding_dim: int = 16,
        frontier_relation_dim: int = 4,
        num_relations: int = 7,
        max_steps: int = 5,
        training_beam_size: int = 2,
        length_penalty_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim + node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_encoder = graph_encoder
        self.scorer = EvidenceNodeScorer(
            hidden_dim=hidden_dim,
            scorer_feature_dim=scorer_feature_dim,
            dropout=dropout,
        )
        decoder_dim = decoder_hidden_dim or hidden_dim
        self.step_embedding = nn.Embedding(max_steps + 1, step_embedding_dim)
        self.frontier_projection = nn.Linear(num_relations * 2, frontier_relation_dim)
        self.first_hop_scorer = _decoder_head(
            hidden_dim * 3 + scorer_feature_dim + 1,
            decoder_dim,
            dropout,
        )
        self.subsequent_hop_scorer = _decoder_head(
            hidden_dim * 5
            + scorer_feature_dim
            + 1
            + frontier_relation_dim
            + step_embedding_dim,
            decoder_dim,
            dropout,
        )
        self.stop_scorer = _decoder_head(
            hidden_dim * 3 + step_embedding_dim,
            decoder_dim,
            dropout,
        )
        self.max_steps = max_steps
        self.num_relations = num_relations
        self.training_beam_size = training_beam_size
        self.length_penalty_alpha = length_penalty_alpha

    def forward(self, batch: TrainingBatch) -> Tensor:
        encoded = self.encode_graph(batch)
        return self.base_score(
            node_states=encoded.node_states[batch.sample_node_indices],
            query_states=encoded.node_states[batch.sample_query_indices],
            sample_node_features=batch.sample_node_features,
        )

    def base_score(
        self,
        *,
        node_states: Tensor,
        query_states: Tensor,
        sample_node_features: Tensor,
    ) -> Tensor:
        return self.scorer(
            node_states=node_states,
            query_states=query_states,
            sample_node_features=sample_node_features,
        )

    def encode_graph(self, batch: TrainingBatch) -> EncodedGraphState:
        graph_batch = batch.graph_batch
        h0 = self.input_projection(
            torch.cat([graph_batch.node_embeddings, graph_batch.node_features], dim=1)
        )
        node_states = self.graph_encoder(graph_batch, h0)
        candidate_node_indices = (
            batch.candidate_node_indices
            if batch.candidate_node_indices is not None
            else batch.sample_node_indices
        )
        candidate_query_indices = (
            batch.candidate_query_indices
            if batch.candidate_query_indices is not None
            else batch.sample_query_indices
        )
        candidate_node_features = (
            batch.candidate_node_features
            if batch.candidate_node_features is not None
            else batch.sample_node_features
        )
        candidate_offsets, candidate_ids = _candidate_metadata(batch)
        base_logits = self.base_score(
            node_states=node_states[candidate_node_indices],
            query_states=node_states[candidate_query_indices],
            sample_node_features=candidate_node_features,
        )
        return EncodedGraphState(
            graph_batch=graph_batch,
            node_states=node_states,
            question_node_indices=graph_batch.query_node_indices,
            task_node_offsets=graph_batch.task_node_offsets,
            task_ids=graph_batch.task_ids,
            node_ids_by_task=graph_batch.node_ids_by_task,
            candidate_node_indices=candidate_node_indices,
            candidate_query_indices=candidate_query_indices,
            candidate_node_features=candidate_node_features,
            candidate_task_offsets=candidate_offsets,
            candidate_node_ids_by_task=candidate_ids,
            base_node_logits=base_logits,
        )

    def selected_context(
        self,
        encoded: EncodedGraphState,
        *,
        task_index: int,
        selected_local_indices: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        query_state = encoded.node_states[encoded.question_node_indices[task_index]]
        if not selected_local_indices:
            zero = torch.zeros_like(query_state)
            return zero, zero
        candidates = encoded.candidate_states_for_task(task_index)
        selected = candidates[
            torch.tensor(
                selected_local_indices, dtype=torch.long, device=candidates.device
            )
        ]
        weights = torch.softmax(
            selected @ query_state / (query_state.shape[0] ** 0.5), dim=0
        )
        return torch.sum(selected * weights.unsqueeze(1), dim=0), selected[-1]

    def score_decoder_actions(
        self,
        encoded: EncodedGraphState,
        *,
        task_index: int,
        selected_local_indices: tuple[int, ...],
    ) -> DecoderActionScores:
        start = encoded.candidate_task_offsets[task_index]
        end = encoded.candidate_task_offsets[task_index + 1]
        candidate_states = encoded.candidate_states_for_task(task_index)
        query_state = encoded.node_states[encoded.question_node_indices[task_index]]
        query_states = query_state.unsqueeze(0).expand(candidate_states.shape[0], -1)
        features = encoded.candidate_node_features[start:end]
        base_logits = encoded.base_node_logits[start:end]
        common = torch.cat(
            [
                candidate_states,
                query_states,
                candidate_states * query_states,
                features,
                base_logits.unsqueeze(1),
            ],
            dim=1,
        )
        selected_context, last_state = self.selected_context(
            encoded,
            task_index=task_index,
            selected_local_indices=selected_local_indices,
        )
        step_index = min(len(selected_local_indices), self.max_steps)
        step = self.step_embedding(
            torch.tensor(step_index, dtype=torch.long, device=candidate_states.device)
        )
        if not selected_local_indices:
            candidate_logits = self.first_hop_scorer(common).squeeze(-1)
        else:
            frontier = self.frontier_projection(
                _frontier_features(
                    encoded,
                    task_index=task_index,
                    selected_local_indices=selected_local_indices,
                    num_relations=self.num_relations,
                )
            )
            expanded = torch.cat(
                [
                    common,
                    selected_context.unsqueeze(0).expand(candidate_states.shape[0], -1),
                    last_state.unsqueeze(0).expand(candidate_states.shape[0], -1),
                    frontier,
                    step.unsqueeze(0).expand(candidate_states.shape[0], -1),
                ],
                dim=1,
            )
            candidate_logits = self.subsequent_hop_scorer(expanded).squeeze(-1)
        stop_input = torch.cat([query_state, selected_context, last_state, step], dim=0)
        stop_logit = self.stop_scorer(stop_input.unsqueeze(0)).squeeze()
        return DecoderActionScores(
            candidate_logits=candidate_logits, stop_logit=stop_logit
        )


def _decoder_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _candidate_metadata(batch: TrainingBatch) -> tuple[list[int], list[list[str]]]:
    if (
        batch.candidate_task_offsets is not None
        and batch.candidate_node_ids_by_task is not None
    ):
        return batch.candidate_task_offsets, batch.candidate_node_ids_by_task
    offsets = [0]
    node_ids_by_task: list[list[str]] = []
    for task_id in batch.graph_batch.task_ids:
        node_ids = [
            node_id
            for sample_task_id, node_id in zip(
                batch.sample_task_ids, batch.sample_node_ids, strict=True
            )
            if sample_task_id == task_id
        ]
        node_ids_by_task.append(node_ids)
        offsets.append(offsets[-1] + len(node_ids))
    return offsets, node_ids_by_task


def _frontier_features(
    encoded: EncodedGraphState,
    *,
    task_index: int,
    selected_local_indices: tuple[int, ...],
    num_relations: int,
) -> Tensor:
    start = encoded.candidate_task_offsets[task_index]
    end = encoded.candidate_task_offsets[task_index + 1]
    candidate_globals = encoded.candidate_node_indices[start:end]
    selected_globals = candidate_globals[
        torch.tensor(
            selected_local_indices, dtype=torch.long, device=candidate_globals.device
        )
    ]
    features = torch.zeros(
        (end - start, num_relations * 2),
        dtype=encoded.node_states.dtype,
        device=encoded.node_states.device,
    )
    edge_index = encoded.graph_batch.edge_index
    if edge_index.numel() == 0:
        return features
    global_to_local = torch.full(
        (encoded.node_states.shape[0],),
        -1,
        dtype=torch.long,
        device=encoded.node_states.device,
    )
    global_to_local[candidate_globals] = torch.arange(
        candidate_globals.shape[0], device=candidate_globals.device
    )
    source, target = edge_index[0], edge_index[1]
    relations = encoded.graph_batch.relation_ids
    weights = encoded.graph_batch.edge_weights

    forward_mask = torch.isin(source, selected_globals) & (global_to_local[target] >= 0)
    features.index_put_(
        (global_to_local[target[forward_mask]], relations[forward_mask]),
        weights[forward_mask],
        accumulate=True,
    )
    reverse_mask = torch.isin(target, selected_globals) & (global_to_local[source] >= 0)
    features.index_put_(
        (
            global_to_local[source[reverse_mask]],
            num_relations + relations[reverse_mask],
        ),
        weights[reverse_mask],
        accumulate=True,
    )
    return features


def _relation_degree_norm(
    *, target_indices: Tensor, relation_ids: Tensor, num_nodes: int
) -> Tensor:
    norm = torch.empty(
        target_indices.shape[0], dtype=torch.float32, device=target_indices.device
    )
    for relation_id in torch.unique(relation_ids):
        mask = relation_ids == relation_id
        relation_targets = target_indices[mask]
        counts = torch.zeros(
            num_nodes, dtype=torch.float32, device=target_indices.device
        )
        counts.index_add_(
            0, relation_targets, torch.ones_like(relation_targets, dtype=torch.float32)
        )
        norm[mask] = 1.0 / counts[relation_targets].clamp_min(1.0)
    return norm
