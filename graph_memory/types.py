from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, TypedDict
from typing_extensions import NotRequired

TaskId = str
NodeId = str
MethodName = str
Score = float
JsonObject: TypeAlias = Mapping[str, "JsonValue"]
JsonArray: TypeAlias = Sequence["JsonValue"]
JsonValue: TypeAlias = str | int | float | bool | None | JsonArray | JsonObject

NodeType = Literal["question", "document_sentence"]
EdgeType = Literal["sequential", "query_overlap", "entity_overlap", "bridge"]
RetrievalMethod = Literal["bm25", "dense", "bm25_graph_rerank", "dense_graph_rerank"]

SUPPORTED_METHODS: set[str] = {
    "bm25",
    "dense",
    "bm25_graph_rerank",
    "dense_graph_rerank",
}

ALLOWED_NODE_TYPES: set[str] = {"question", "document_sentence"}
ALLOWED_EDGE_TYPES: set[str] = {"sequential", "query_overlap", "entity_overlap", "bridge"}
NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES: set[str] = {"sequential", "entity_overlap", "bridge"}


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


class GraphRerankConfigRecord(TypedDict, total=False):
    lambda_init: float
    lambda_query: float
    lambda_neighbor: float
    lambda_bridge: float
    lambda_path: float
    seed_top_s: int
    max_hops: int
    neighbor_type_weights: dict[str, float]
    # Deprecated compatibility input. New artifacts should write neighbor_type_weights.
    type_weights: dict[str, float]


class TuningCandidateRow(MetricRow):
    config: GraphRerankConfigRecord


class GraphStatistics(TypedDict, total=False):
    num_graphs: int
    avg_nodes: float
    avg_edges: float
    edge_counts_by_type: dict[str, int]
    isolated_memory_nodes: int
    split: str
    graph_config: JsonObject


class RunSummary(TypedDict, total=False):
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
    error: str


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


def default_neighbor_type_weights() -> dict[str, float]:
    return {
        "sequential": 0.3,
        "entity_overlap": 0.7,
        "bridge": 1.0,
    }


def default_type_weights() -> dict[str, float]:
    """Deprecated alias for historical callers; returns neighbor type weights."""

    return default_neighbor_type_weights()


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


def graph_rerank_config_from_value(
    value: GraphRerankConfig | Mapping[str, object] | None,
) -> GraphRerankConfig:
    if value is None:
        raise ValueError("Graph rerank methods require graph_config.")
    if isinstance(value, GraphRerankConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Graph rerank config must be a GraphRerankConfig or mapping.")

    kwargs = dict(value)
    deprecated_type_weights = kwargs.pop("type_weights", None)
    if "neighbor_type_weights" not in kwargs and isinstance(deprecated_type_weights, Mapping):
        kwargs["neighbor_type_weights"] = {
            str(edge_type): float(weight)
            for edge_type, weight in deprecated_type_weights.items()
            if str(edge_type) != "query_overlap"
        }
    if isinstance(kwargs.get("neighbor_type_weights"), Mapping):
        kwargs["neighbor_type_weights"] = {
            str(edge_type): float(weight)
            for edge_type, weight in dict(kwargs["neighbor_type_weights"]).items()
        }
    return GraphRerankConfig(**kwargs)


@dataclass(frozen=True)
class RerankResult:
    ranked_nodes: list[RankedNode]
    retrieved_subgraph: RetrievedSubgraph
    score_breakdown: ScoreBreakdown | None = None


class Retriever(Protocol):
    method_name: str

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        ...
