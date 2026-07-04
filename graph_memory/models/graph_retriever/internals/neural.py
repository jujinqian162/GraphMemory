from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from graph_memory.models.graph_retriever.internals.contracts import GraphBatch, TrainingBatch


class GraphEncoder(Protocol):
    """
    Replaceable graph encoder over batch-flattened node states.
    基于 batch-flattened node state 的可替换图编码器。

    Methods / 方法:
    - forward: Return encoded node states with the same first dimension as input node states.
      forward：返回编码后的 node states，第一维必须与输入 node states 一致。
    """

    def forward(self, batch: GraphBatch, node_states: Tensor) -> Tensor:
        ...


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

    def forward(self, source_states: Tensor, relation_ids: Tensor) -> Tensor:
        ...


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

    def __init__(self, *, hidden_dim: int, message_transform: MessageTransform, dropout: float) -> None:
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


class GatedRGCNGraphEncoder(nn.Module):
    """
    R-GCN encoder wrapper that multiplies message edge weights by learned gates.
    用 learned gates 缩放 message edge weights 的 R-GCN encoder 包装器。
    """

    def __init__(self, *, base_encoder: RGCNGraphEncoder) -> None:
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, batch: GraphBatch, node_states: Tensor, *, edge_gates: Tensor) -> Tensor:
        if edge_gates.shape != batch.edge_weights.shape:
            raise ValueError("edge_gates must have the same shape as edge_weights.")
        gated_batch = GraphBatch(
            node_embeddings=batch.node_embeddings,
            node_features=batch.node_features,
            edge_index=batch.edge_index,
            relation_ids=batch.relation_ids,
            edge_weights=batch.edge_weights * edge_gates,
            query_node_indices=batch.query_node_indices,
            task_node_offsets=batch.task_node_offsets,
            task_ids=batch.task_ids,
            node_ids_by_task=batch.node_ids_by_task,
        )
        return self.base_encoder.forward(gated_batch, node_states)


class EvidenceNodeScorer(nn.Module):
    """
    MLP scorer that maps memory/query node states to one evidence logit per sample.
    将 memory/query node state 映射为每个 sample 一个 evidence logit 的 MLP scorer。

    Fields / 字段:
    - network: MLP over `[h_node, h_query, h_node * h_query, sample_node_features]`.
      network：作用在 `[h_node, h_query, h_node * h_query, sample_node_features]` 上的 MLP。
    """

    def __init__(self, *, hidden_dim: int, scorer_feature_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = hidden_dim * 3 + scorer_feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, *, node_states: Tensor, query_states: Tensor, sample_node_features: Tensor) -> Tensor:
        scorer_input = torch.cat([node_states, query_states, node_states * query_states, sample_node_features], dim=1)
        return self.network(scorer_input).squeeze(-1)


class EdgeGateScorer(nn.Module):
    """
    MLP scorer that maps edge features to one gate logit per message edge.
    将 edge features 映射为每条 message edge 一个 gate logit 的 MLP。
    """

    def __init__(self, *, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features: Tensor) -> Tensor:
        if edge_features.ndim != 2:
            raise ValueError("edge_features must be a 2D tensor.")
        return self.network(edge_features).squeeze(-1)


@dataclass(frozen=True)
class EvidenceScoringOutput:
    node_logits: Tensor
    edge_gate_logits: Tensor | None = None
    edge_gates: Tensor | None = None


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
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim + node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_encoder = graph_encoder
        self.scorer = EvidenceNodeScorer(hidden_dim=hidden_dim, scorer_feature_dim=scorer_feature_dim, dropout=dropout)

    def forward(self, batch: TrainingBatch) -> Tensor:
        graph_batch = batch.graph_batch
        h0 = self.input_projection(torch.cat([graph_batch.node_embeddings, graph_batch.node_features], dim=1))
        h = self.graph_encoder.forward(graph_batch, h0)
        node_states = h[batch.sample_node_indices]
        query_states = h[batch.sample_query_indices]
        return self.scorer(
            node_states=node_states,
            query_states=query_states,
            sample_node_features=batch.sample_node_features,
        )


class LearnedGraphEvidenceScoringModel(nn.Module):
    """
    Evidence scorer that learns edge gates before R-GCN message passing.
    在 R-GCN message passing 前学习 edge gates 的 evidence scorer。
    """

    def __init__(
        self,
        *,
        encoder_dim: int,
        node_feature_dim: int,
        hidden_dim: int,
        graph_encoder: GatedRGCNGraphEncoder,
        scorer_feature_dim: int,
        edge_feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_dim + node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_encoder = graph_encoder
        self.edge_gate_scorer = EdgeGateScorer(feature_dim=edge_feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.scorer = EvidenceNodeScorer(hidden_dim=hidden_dim, scorer_feature_dim=scorer_feature_dim, dropout=dropout)

    def forward(self, batch: TrainingBatch) -> EvidenceScoringOutput:
        if batch.learned_edges is None:
            raise ValueError("Learned graph scoring requires learned edge tensors.")
        graph_batch = batch.graph_batch
        h0 = self.input_projection(torch.cat([graph_batch.node_embeddings, graph_batch.node_features], dim=1))
        edge_gate_logits = self.edge_gate_scorer(batch.learned_edges.edge_features)
        edge_gates = torch.sigmoid(edge_gate_logits)
        h = self.graph_encoder.forward(graph_batch, h0, edge_gates=edge_gates)
        node_states = h[batch.sample_node_indices]
        query_states = h[batch.sample_query_indices]
        node_logits = self.scorer(
            node_states=node_states,
            query_states=query_states,
            sample_node_features=batch.sample_node_features,
        )
        return EvidenceScoringOutput(
            node_logits=node_logits,
            edge_gate_logits=edge_gate_logits,
            edge_gates=edge_gates,
        )


def _relation_degree_norm(*, target_indices: Tensor, relation_ids: Tensor, num_nodes: int) -> Tensor:
    norm = torch.empty(target_indices.shape[0], dtype=torch.float32, device=target_indices.device)
    for relation_id in torch.unique(relation_ids):
        mask = relation_ids == relation_id
        relation_targets = target_indices[mask]
        counts = torch.zeros(num_nodes, dtype=torch.float32, device=target_indices.device)
        counts.index_add_(0, relation_targets, torch.ones_like(relation_targets, dtype=torch.float32))
        norm[mask] = 1.0 / counts[relation_targets].clamp_min(1.0)
    return norm
