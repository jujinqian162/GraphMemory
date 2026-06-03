from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from graph_memory.contracts.common import (
    ALLOWED_EDGE_TYPES,
    ALLOWED_NODE_TYPES,
    NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES,
    NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES,
    TRAIN_PAIR_SAMPLE_TYPES,
    EdgeType,
    JsonArray,
    JsonObject,
    JsonValue,
    MethodName,
    NodeId,
    NodeType,
    Score,
    TaskId,
    TrainPairSampleType,
)
from graph_memory.contracts.graphs import GraphEdge, GraphMemoryNode, GraphNode, MemoryGraph, QuestionNode
from graph_memory.contracts.metrics import FailureCase, MetricRow, MetricTableRow, MetricValue, TaskMetricRow
from graph_memory.contracts.observability import GraphStatistics, RankedNodeDebugRecord, RunSummary, ScoreDebugRecord
from graph_memory.contracts.ranking import RankedNodeRecord, RankedResult, RetrievedSubgraph
from graph_memory.contracts.tasks import CombinedMemoryTask, MemoryItem, MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairBuildSummary, TrainPairRecord
from graph_memory.graphs.config import GraphBuildConfig

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

@dataclass(frozen=True)
class RankedNode:
    node_id: NodeId
    score: Score


@dataclass(frozen=True)
class ScoreComponents:
    initial: float
    query: float = 0.0
    neighbor: float = 0.0
    bridge: float = 0.0
    path: float = 0.0
    final: float = 0.0


ScoreBreakdown = dict[NodeId, ScoreComponents]


class GraphRerankConfigRecord(TypedDict):
    lambda_init: float
    lambda_query: float
    lambda_neighbor: float
    lambda_bridge: float
    lambda_path: float
    seed_top_s: int
    max_hops: int
    neighbor_type_weights: dict[str, float]


class TuningCandidateRow(MetricRow):
    config: GraphRerankConfigRecord


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """
    Configuration for deterministic train pair negative sampling.
    确定性训练 pair 负采样配置。

    Fields / 字段:
    - random_seed: Seed used by random negative sampling and tie-breaking.
      random_seed：随机负采样和 tie-breaking 使用的种子。
    - easy_random_per_positive: Number of easy random negatives per positive node.
      easy_random_per_positive：每个正例对应的 easy random 负例数量。
    - hard_bm25_per_positive: Number of hard BM25 negatives per positive node.
      hard_bm25_per_positive：每个正例对应的 hard BM25 负例数量。
    - hard_dense_per_positive: Number of hard dense negatives per positive node.
      hard_dense_per_positive：每个正例对应的 hard dense 负例数量。
    - hard_graph_neighbor_per_positive: Number of graph-neighbor negatives per positive node.
      hard_graph_neighbor_per_positive：每个正例对应的 graph-neighbor 负例数量。
    - hard_pool_size: Top-ranked non-gold candidate pool size for hard retriever negatives.
      hard_pool_size：hard retriever 负例采样时使用的非 gold top-ranked 候选池大小。
    """

    random_seed: int = 13
    easy_random_per_positive: int = 2
    hard_bm25_per_positive: int = 2
    hard_dense_per_positive: int = 2
    hard_graph_neighbor_per_positive: int = 1
    hard_pool_size: int = 30


@dataclass(frozen=True)
class NodeFeatureConfig:
    """
    Ordered numeric node feature configuration.
    有序的节点数值特征配置。

    Fields / 字段:
    - node_feature_names: Ordered features concatenated with text embeddings before input projection.
      node_feature_names：input projection 前与文本 embedding 拼接的有序特征名。
    - scorer_feature_names: Ordered direct numeric features passed to the evidence scorer.
      scorer_feature_names：直接传入 evidence scorer 的有序数值特征名。
    """

    node_feature_names: tuple[str, ...] = ("seed_score", "seed_rank_percentile", "is_question_node")
    scorer_feature_names: tuple[str, ...] = ("seed_score", "seed_rank_percentile")


@dataclass(frozen=True)
class TrainableModelConfig:
    """
    Minimal model reconstruction config saved in every trainable checkpoint.
    每个可训练 checkpoint 中保存的最小模型重建配置。

    Fields / 字段:
    - method_name: Public retrieval method name.
      method_name：公开检索方法名。
    - encoder_model: Frozen text encoder model name.
      encoder_model：冻结文本 encoder 的模型名。
    - encoder_dim: Frozen text embedding dimension used by the model input projection.
      encoder_dim：模型 input projection 使用的冻结文本 embedding 维度。
    - query_prefix: Prefix applied to query text before encoding.
      query_prefix：编码 query 文本前添加的前缀。
    - passage_prefix: Prefix applied to memory text before encoding.
      passage_prefix：编码 memory 文本前添加的前缀。
    - hidden_dim: Hidden dimension used by graph encoder and scorer.
      hidden_dim：graph encoder 和 scorer 使用的隐藏维度。
    - num_layers: Number of R-GCN layers; 0 means identity graph encoder.
      num_layers：R-GCN 层数；0 表示 identity graph encoder。
    - dropout: Dropout probability.
      dropout：dropout 概率。
    - feature_config: Ordered node and scorer feature names.
      feature_config：有序的 node 和 scorer 特征名。
    - relation_vocab: Ordered relation names used by relation ids.
      relation_vocab：relation id 使用的有序 relation 名称。
    - graph_encoder_type: Graph encoder component name, such as `rgcn` or `identity`.
      graph_encoder_type：graph encoder 组件名，例如 `rgcn` 或 `identity`。
    - message_transform_type: Relation transform component name, such as `typed` or `shared`.
      message_transform_type：relation transform 组件名，例如 `typed` 或 `shared`。
    - edge_weight_policy: Edge weight policy name, such as `artifact` or `uniform`.
      edge_weight_policy：edge weight policy 名称，例如 `artifact` 或 `uniform`。
    - enabled_edge_types: Ordered graph artifact edge types enabled during tensorization.
      enabled_edge_types：tensorization 时启用的有序 graph artifact edge type。
    - ablation_name: Canonical experiment or ablation name.
      ablation_name：规范化的实验或 ablation 名称。
    """

    method_name: MethodName
    encoder_model: str
    encoder_dim: int
    query_prefix: str
    passage_prefix: str
    hidden_dim: int
    num_layers: int
    dropout: float
    feature_config: NodeFeatureConfig
    relation_vocab: tuple[str, ...]
    graph_encoder_type: str
    message_transform_type: str
    edge_weight_policy: str
    enabled_edge_types: tuple[str, ...]
    ablation_name: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "encoder_model": self.encoder_model,
            "encoder_dim": self.encoder_dim,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "feature_config": {
                "node_feature_names": list(self.feature_config.node_feature_names),
                "scorer_feature_names": list(self.feature_config.scorer_feature_names),
            },
            "relation_vocab": list(self.relation_vocab),
            "graph_encoder_type": self.graph_encoder_type,
            "message_transform_type": self.message_transform_type,
            "edge_weight_policy": self.edge_weight_policy,
            "enabled_edge_types": list(self.enabled_edge_types),
            "ablation_name": self.ablation_name,
        }


@dataclass(frozen=True)
class TrainableTrainingConfig:
    """
    Minimal training config needed to resume or audit a trainable run.
    用于恢复或审计可训练运行的最小训练配置。

    Fields / 字段:
    - optimizer_name: Optimizer name, default `AdamW`.
      optimizer_name：优化器名称，默认 `AdamW`。
    - learning_rate: Graph/scorer learning rate.
      learning_rate：graph/scorer 学习率。
    - batch_size: Number of task graphs per training batch.
      batch_size：每个 training batch 中的 task graph 数量。
    - max_grad_norm: Gradient clipping maximum norm.
      max_grad_norm：梯度裁剪最大 norm。
    - random_seed: Run-level random seed.
      random_seed：运行级随机种子。
    - pos_weight_enabled: Whether BCE positive weighting was enabled.
      pos_weight_enabled：是否启用 BCE 正例权重。
    - epochs: Number of training epochs.
      epochs：训练 epoch 数量。
    """

    optimizer_name: str = "AdamW"
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_grad_norm: float = 1.0
    random_seed: int = 13
    pos_weight_enabled: bool = False
    epochs: int = 1

    def to_json_dict(self) -> dict[str, object]:
        return {
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "random_seed": self.random_seed,
            "pos_weight_enabled": self.pos_weight_enabled,
            "epochs": self.epochs,
        }


@dataclass(frozen=True)
class SeedSignal:
    """
    Frozen seed retrieval signal for one memory node.
    一个 memory node 的冻结初始检索信号。

    Fields / 字段:
    - node_id: Memory node id receiving this seed signal.
      node_id：该 seed signal 对应的 memory node id。
    - score: Raw seed retriever score.
      score：seed retriever 原始分数。
    - rank: One-based rank after descending score and ascending node id tie-break.
      rank：从 1 开始的排名；按 score 降序、node id 升序打破平局。
    - rank_percentile: Rank percentile in [0, 1], where 0 means best and 1 means worst.
      rank_percentile：范围 [0, 1] 的排名百分位，0 表示最好，1 表示最差。
    """

    node_id: NodeId
    score: float
    rank: int
    rank_percentile: float


@dataclass(frozen=True)
class GraphBatch:
    """
    Tensorized batch of one or more task graphs for message passing.
    一个或多个 task graph 的 message passing 张量化 batch。

    Fields / 字段:
    - node_embeddings: Tensor `[total_nodes, encoder_dim]` with frozen query and memory text embeddings.
      node_embeddings：形状为 `[total_nodes, encoder_dim]` 的冻结 query 和 memory 文本 embedding。
    - node_features: Tensor `[total_nodes, node_feature_dim]` with ordered numeric node features.
      node_features：形状为 `[total_nodes, node_feature_dim]` 的有序节点数值特征。
    - edge_index: Long tensor `[2, num_message_edges]`; row 0 is source, row 1 is target.
      edge_index：Long tensor，形状为 `[2, num_message_edges]`；第 0 行是 source，第 1 行是 target。
    - relation_ids: Long tensor `[num_message_edges]` indexing the saved relation vocab.
      relation_ids：Long tensor，形状为 `[num_message_edges]`，索引保存的 relation vocab。
    - edge_weights: Float tensor `[num_message_edges]` with tensorizer-produced message edge weights.
      edge_weights：Float tensor，形状为 `[num_message_edges]`，来自 tensorizer 的 message edge 权重。
    - query_node_indices: Long tensor `[num_tasks]` containing each task question node global index.
      query_node_indices：Long tensor，形状为 `[num_tasks]`，包含每个 task 的问题节点全局 index。
    - task_node_offsets: Python list of length `num_tasks + 1`; start inclusive, end exclusive.
      task_node_offsets：长度为 `num_tasks + 1` 的 Python list；起点包含，终点不包含。
    - task_ids: Task ids in the same order as `task_node_offsets`.
      task_ids：与 `task_node_offsets` 顺序一致的 task id。
    - node_ids_by_task: Per-task node ids in local tensorization order; `q` must be present.
      node_ids_by_task：每个 task 内按本地张量化顺序排列的 node id；必须包含 `q`。
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

    Fields / 字段:
    - graph_batch: Shared graph tensor batch used once for message passing.
      graph_batch：共享的 graph tensor batch，用于一次 message passing。
    - sample_node_indices: Long tensor `[num_samples]` with batch-flattened memory node indexes.
      sample_node_indices：Long tensor，形状为 `[num_samples]`，包含 batch-flattened memory node index。
    - sample_query_indices: Long tensor `[num_samples]` with matching question node indexes.
      sample_query_indices：Long tensor，形状为 `[num_samples]`，包含匹配的问题节点 index。
    - sample_node_features: Float tensor `[num_samples, scorer_feature_dim]` for direct scorer features.
      sample_node_features：Float tensor，形状为 `[num_samples, scorer_feature_dim]`，用于 scorer 直接特征。
    - labels: Float tensor `[num_samples]` with binary labels.
      labels：Float tensor，形状为 `[num_samples]`，包含二分类标签。
    - sample_task_ids: Task id for each sample, used for debug and metric grouping.
      sample_task_ids：每个 sample 对应的 task id，用于 debug 和 metric 分组。
    - sample_node_ids: Memory node id for each sample, used for debug and validation.
      sample_node_ids：每个 sample 对应的 memory node id，用于 debug 和 validation。
    - sample_types: Sampling source for each sample.
      sample_types：每个 sample 的采样来源。
    """

    graph_batch: GraphBatch
    sample_node_indices: Tensor
    sample_query_indices: Tensor
    sample_node_features: Tensor
    labels: Tensor
    sample_task_ids: list[TaskId]
    sample_node_ids: list[NodeId]
    sample_types: list[TrainPairSampleType]


def default_neighbor_type_weights() -> dict[str, float]:
    return {
        "sequential": 0.3,
        "entity_overlap": 0.7,
        "bridge": 1.0,
    }


@dataclass(frozen=True)
class GraphRerankConfig:
    lambda_init: float = 1.0
    lambda_query: float = 0.1
    lambda_neighbor: float = 0.2
    lambda_bridge: float = 0.1
    lambda_path: float = 0.0
    seed_top_s: int = 30
    max_hops: int = 2
    neighbor_type_weights: dict[str, float] = field(default_factory=default_neighbor_type_weights)


@dataclass(frozen=True)
class RerankResult:
    ranked_nodes: list[RankedNode]
    retrieved_subgraph: RetrievedSubgraph
    score_breakdown: ScoreBreakdown | None = None


class Retriever(Protocol):
    @property
    def method_name(self) -> str:
        ...

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        ...


__all__ = [
    "ALLOWED_EDGE_TYPES",
    "ALLOWED_NODE_TYPES",
    "CombinedMemoryTask",
    "DenseConfig",
    "EdgeType",
    "FailureCase",
    "GraphBatch",
    "GraphBuildConfig",
    "GraphEdge",
    "GraphMemoryNode",
    "GraphNode",
    "GraphRerankConfig",
    "GraphRerankConfigRecord",
    "GraphStatistics",
    "JsonArray",
    "JsonObject",
    "JsonValue",
    "MemoryGraph",
    "MemoryItem",
    "MemoryTaskInput",
    "MemoryTaskLabels",
    "MethodName",
    "NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES",
    "NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES",
    "NegativeSamplingConfig",
    "NodeFeatureConfig",
    "NodeId",
    "NodeType",
    "QuestionNode",
    "RankedNode",
    "RankedNodeDebugRecord",
    "RankedNodeRecord",
    "RankedResult",
    "RerankResult",
    "RetrievedSubgraph",
    "Retriever",
    "RunSummary",
    "Score",
    "ScoreBreakdown",
    "ScoreComponents",
    "ScoreDebugRecord",
    "SeedSignal",
    "TRAIN_PAIR_SAMPLE_TYPES",
    "TaskId",
    "TrainPairBuildSummary",
    "TrainPairRecord",
    "TrainPairSampleType",
    "TrainableModelConfig",
    "TrainableTrainingConfig",
    "TrainingBatch",
    "TuningCandidateRow",
    "MetricRow",
    "MetricTableRow",
    "MetricValue",
    "TaskMetricRow",
]
