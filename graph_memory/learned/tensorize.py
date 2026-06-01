from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

from graph_memory.types import ALLOWED_EDGE_TYPES, GraphEdge, MemoryGraph

DEFAULT_RELATION_VOCAB: tuple[str, ...] = (
    "query_overlap_forward",
    "sequential_forward",
    "sequential_reverse",
    "entity_overlap_forward",
    "entity_overlap_reverse",
    "bridge_forward",
    "bridge_reverse",
)


class EdgeWeightPolicy(Protocol):
    """
    Replaceable policy for message edge weights during tensorization.
    tensorization 阶段可替换的 message edge 权重策略。

    Methods / 方法:
    - weight: Return the float message edge weight for one graph edge.
      weight：返回一个 graph edge 对应的 float message edge 权重。
    """

    def weight(self, edge: GraphEdge) -> float:
        ...


@dataclass(frozen=True)
class ArtifactEdgeWeightPolicy:
    """
    Edge weight policy that preserves graph artifact weights.
    保留 graph artifact 权重的 edge weight 策略。

    Methods / 方法:
    - weight: Return `edge["weight"]` as a float.
      weight：以 float 返回 `edge["weight"]`。
    """

    def weight(self, edge: GraphEdge) -> float:
        return float(edge["weight"])


@dataclass(frozen=True)
class UniformEdgeWeightPolicy:
    """
    Edge weight policy that replaces every message edge weight with 1.0.
    将所有 message edge 权重替换为 1.0 的 edge weight 策略。

    Methods / 方法:
    - weight: Return `1.0` regardless of artifact edge weight.
      weight：无论 artifact edge weight 是多少，都返回 `1.0`。
    """

    def weight(self, edge: GraphEdge) -> float:
        return 1.0


@dataclass(frozen=True)
class MessageEdgeTensors:
    """
    Tensorized message edge arrays for one graph.
    单个 graph 的 message edge 张量。

    Fields / 字段:
    - edge_index: Long tensor `[2, num_message_edges]`; row 0 is source and row 1 is target.
      edge_index：Long tensor，形状为 `[2, num_message_edges]`；第 0 行是 source，第 1 行是 target。
    - relation_ids: Long tensor `[num_message_edges]` indexing the relation vocab.
      relation_ids：Long tensor，形状为 `[num_message_edges]`，索引 relation vocab。
    - edge_weights: Float tensor `[num_message_edges]` after applying the edge weight policy.
      edge_weights：Float tensor，形状为 `[num_message_edges]`，应用 edge weight policy 后的权重。
    """

    edge_index: Tensor
    relation_ids: Tensor
    edge_weights: Tensor


@dataclass(frozen=True)
class EdgeTensorizer:
    """
    Converts graph artifact edges into directed message-passing tensors.
    将 graph artifact edge 转换为有向 message-passing 张量。

    Fields / 字段:
    - relation_vocab: Ordered relation names used by relation ids.
      relation_vocab：relation id 使用的有序 relation 名称。
    - enabled_edge_types: Graph artifact edge types allowed during tensorization.
      enabled_edge_types：tensorization 期间允许使用的 graph artifact edge type。
    - edge_weight_policy: Policy that maps graph edge weights to message edge weights.
      edge_weight_policy：将 graph edge weight 映射为 message edge weight 的策略。
    """

    relation_vocab: tuple[str, ...] = DEFAULT_RELATION_VOCAB
    enabled_edge_types: frozenset[str] = field(default_factory=lambda: frozenset(ALLOWED_EDGE_TYPES))
    edge_weight_policy: EdgeWeightPolicy = field(default_factory=ArtifactEdgeWeightPolicy)

    def tensorize_edges(self, graph: MemoryGraph) -> MessageEdgeTensors:
        """
        Tensorize one graph's enabled edges into message edge tensors.
        将单个 graph 中启用的边张量化为 message edge tensors。
        """

        node_index_by_id = {node["id"]: index for index, node in enumerate(graph["nodes"])}
        relation_id_by_name = {relation_name: index for index, relation_name in enumerate(self.relation_vocab)}
        sources: list[int] = []
        targets: list[int] = []
        relation_ids: list[int] = []
        edge_weights: list[float] = []

        for edge in graph["edges"]:
            edge_type = edge["edge_type"]
            if edge_type not in self.enabled_edge_types:
                continue
            source = _node_index(node_index_by_id, edge["source"])
            target = _node_index(node_index_by_id, edge["target"])
            weight = self.edge_weight_policy.weight(edge)
            self._append_message_edge(
                sources,
                targets,
                relation_ids,
                edge_weights,
                source=source,
                target=target,
                relation_name=f"{edge_type}_forward",
                weight=weight,
                relation_id_by_name=relation_id_by_name,
            )
            if not edge["directed"]:
                self._append_message_edge(
                    sources,
                    targets,
                    relation_ids,
                    edge_weights,
                    source=target,
                    target=source,
                    relation_name=f"{edge_type}_reverse",
                    weight=weight,
                    relation_id_by_name=relation_id_by_name,
                )

        if sources:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return MessageEdgeTensors(
            edge_index=edge_index,
            relation_ids=torch.tensor(relation_ids, dtype=torch.long),
            edge_weights=torch.tensor(edge_weights, dtype=torch.float32),
        )

    def _append_message_edge(
        self,
        sources: list[int],
        targets: list[int],
        relation_ids: list[int],
        edge_weights: list[float],
        *,
        source: int,
        target: int,
        relation_name: str,
        weight: float,
        relation_id_by_name: dict[str, int],
    ) -> None:
        if relation_name not in relation_id_by_name:
            raise ValueError(f"Relation {relation_name} is not present in relation_vocab.")
        sources.append(source)
        targets.append(target)
        relation_ids.append(relation_id_by_name[relation_name])
        edge_weights.append(float(weight))


def model_visible_graph(graph: MemoryGraph, enabled_edge_types: frozenset[str]) -> MemoryGraph:
    """Return the graph view visible to one trained model."""

    return {
        **graph,
        "edges": [
            edge
            for edge in graph.get("edges", [])
            if edge.get("edge_type") in enabled_edge_types
        ],
    }


def _node_index(node_index_by_id: dict[str, int], node_id: str) -> int:
    if node_id not in node_index_by_id:
        raise ValueError(f"Graph edge references missing node_id={node_id}.")
    return node_index_by_id[node_id]
