from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

TaskId = str
NodeId = str
MethodName = str
Score = float
JsonObject: TypeAlias = Mapping[str, "JsonValue"]
JsonArray: TypeAlias = Sequence["JsonValue"]
JsonValue: TypeAlias = str | int | float | bool | None | JsonArray | JsonObject

NodeType = Literal["question", "document_sentence"]
EdgeType = Literal["sequential", "query_overlap", "entity_overlap", "bridge"]
TrainPairSampleType = Literal["positive", "easy_random", "hard_bm25", "hard_dense", "hard_graph_neighbor"]

ALLOWED_NODE_TYPES: set[str] = {"question", "document_sentence"}
ALLOWED_EDGE_TYPES: set[str] = {"sequential", "query_overlap", "entity_overlap", "bridge"}
NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES: set[str] = {"sequential", "entity_overlap", "bridge"}
TRAIN_PAIR_SAMPLE_TYPES: set[str] = {"positive", "easy_random", "hard_bm25", "hard_dense", "hard_graph_neighbor"}
NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES: set[str] = TRAIN_PAIR_SAMPLE_TYPES - {"positive"}


class MemoryItem(TypedDict):
    id: NodeId
    node_type: Literal["document_sentence"]
    text: str
    source: str
    sentence_id: int
    position: int


class MemoryTaskInput(TypedDict):
    task_id: TaskId
    query: str
    memory_items: list[MemoryItem]
    metadata: NotRequired[dict[str, object]]
    debug: NotRequired[dict[str, object]]


class MemoryTaskLabels(TypedDict):
    task_id: TaskId
    gold_answer: str
    gold_evidence_nodes: list[NodeId]
    gold_dependency_edges: list[list[str]]
    metadata: NotRequired[dict[str, object]]
    debug: NotRequired[dict[str, object]]


class CombinedMemoryTask(MemoryTaskInput, MemoryTaskLabels):
    """Compatibility-only artifact shape containing input and label fields."""


class QuestionNode(TypedDict):
    id: Literal["q"]
    node_type: Literal["question"]
    text: str


class GraphMemoryNode(MemoryItem):
    pass


GraphNode = QuestionNode | GraphMemoryNode


class GraphEdge(TypedDict):
    source: str
    target: str
    edge_type: EdgeType
    weight: float
    directed: bool


class MemoryGraph(TypedDict):
    task_id: TaskId
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: NotRequired[dict[str, object]]
    debug: NotRequired[dict[str, object]]


class RankedNodeRecord(TypedDict):
    node_id: NodeId
    score: float


class RetrievedSubgraph(TypedDict):
    nodes: list[NodeId]
    edges: list[GraphEdge]


class RankedResult(TypedDict):
    task_id: TaskId
    method: MethodName
    ranked_nodes: list[RankedNodeRecord]
    retrieved_subgraph: RetrievedSubgraph
    latency_ms: float
    input_tokens: int
    metadata: NotRequired[dict[str, object]]
    debug: NotRequired[dict[str, object]]


class TrainPairRecord(TypedDict):
    """
    One training pair artifact row for a query-node supervision example.
    一个 query-node 监督样本对应的训练 pair artifact 行。

    Fields / 字段:
    - task_id: Task join key matching memory task, label, and graph artifacts.
      task_id：任务 join key，必须匹配 memory task、label 和 graph artifact。
    - node_id: Memory node id being supervised; must not be the question node `q`.
      node_id：被监督的 memory node id；不能是问题节点 `q`。
    - label: Binary evidence label, where 1 means gold evidence and 0 means sampled negative.
      label：二分类 evidence 标签，1 表示 gold evidence，0 表示采样负例。
    - sample_type: Sampling source used to create this row.
      sample_type：生成该样本行时使用的采样来源。
    """

    task_id: TaskId
    node_id: NodeId
    label: Literal[0, 1]
    sample_type: TrainPairSampleType


class TrainPairBuildSummary(TypedDict):
    """
    Summary record written beside train pair artifacts for reproducibility.
    写在 train pair artifact 旁边、用于复现性的汇总记录。

    Fields / 字段:
    - positive_count: Number of positive rows.
      positive_count：正例行数。
    - negative_count_by_type: Negative row counts grouped by sample type.
      negative_count_by_type：按 sample type 分组的负例行数。
    - avg_positive_per_task: Average positive rows per task.
      avg_positive_per_task：每个 task 的平均正例行数。
    - avg_negative_per_task: Average negative rows per task.
      avg_negative_per_task：每个 task 的平均负例行数。
    - tasks_with_no_positive: Task ids that had no gold evidence; must be empty.
      tasks_with_no_positive：没有 gold evidence 的 task id；必须为空。
    - sampling_config: Effective negative sampling config.
      sampling_config：实际生效的负采样配置。
    """

    positive_count: int
    negative_count_by_type: dict[str, int]
    avg_positive_per_task: float
    avg_negative_per_task: float
    tasks_with_no_positive: list[TaskId]
    sampling_config: dict[str, object]


MetricValue: TypeAlias = str | float

MetricRow = TypedDict(
    "MetricRow",
    {
        "Method": str,
        "Recall@2": float,
        "Recall@5": float,
        "Recall@10": float,
        "Evidence F1@5": float,
        "Evidence F1@10": float,
        "Full Support@5": float,
        "Full Support@10": float,
        "MRR": float,
        "Connected Evidence Recall@5": float,
        "Connected Evidence Recall@10": float,
        "Query-Evidence Connectivity@10": float,
        "Path Recall@10": MetricValue,
        "Edge Recall@10": MetricValue,
        "Retrieval Latency / Query": float,
        "Index Build Time": float,
        "Graph Construction Time": float,
        "Memory Size": float,
        "Avg Retrieved Nodes": float,
        "Avg Retrieved Edges": float,
    },
)

MetricTableRow: TypeAlias = dict[str, MetricValue]

TaskMetricRow = TypedDict(
    "TaskMetricRow",
    {
        "Recall@2": float,
        "Recall@5": float,
        "Recall@10": float,
        "Evidence F1@5": float,
        "Evidence F1@10": float,
        "Full Support@5": float,
        "Full Support@10": float,
        "MRR": float,
        "Connected Evidence Recall@5": float,
        "Connected Evidence Recall@10": float,
        "Query-Evidence Connectivity@10": float,
        "Retrieval Latency / Query": float,
        "Memory Size": float,
        "Avg Retrieved Nodes": float,
        "Avg Retrieved Edges": float,
    },
)


class FailureCase(TypedDict):
    debug_type: str
    task_id: TaskId
    method: MethodName
    failure_type: str
    gold_evidence_nodes: list[NodeId]
    retrieved_top_k: list[NodeId]
    missing_gold_nodes: list[NodeId]
    connected_gold_in_top_k: bool


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


class GraphStatistics(TypedDict):
    num_graphs: int
    avg_nodes: float
    avg_edges: float
    edge_counts_by_type: dict[str, int]
    isolated_memory_nodes: int
    split: NotRequired[str]
    graph_config: NotRequired[JsonObject]


class RunSummary(TypedDict):
    script: str
    started_at: str
    finished_at: str
    status: str
    effective_config: JsonObject
    inputs: JsonObject
    outputs: JsonObject
    counts: JsonObject
    timings: JsonObject
    environment: dict[str, str]
    notes: list[str]
    error: NotRequired[str]


class RankedNodeDebugRecord(RankedNodeRecord, total=False):
    score_components: ScoreComponents


class ScoreDebugRecord(TypedDict, total=False):
    debug_type: str
    task_id: TaskId
    method: MethodName
    top_k: int
    ranked_nodes: list[RankedNodeDebugRecord]
    split: str
    config_digest: str


@dataclass(frozen=True)
class GraphBuildConfig:
    max_query_overlap: int = 20
    max_entity_neighbors: int = 10
    max_bridge_edges: int = 50
    use_spacy: bool = False


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
