from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NotRequired, Protocol, TypedDict

TaskId = str
NodeId = str
MethodName = str
Score = float

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


class EvaluationRow(TypedDict):
    Method: str
    Recall_at_2: float
    Recall_at_5: float
    Recall_at_10: float
    Evidence_F1_at_5: float
    Evidence_F1_at_10: float
    Full_Support_at_5: float
    Full_Support_at_10: float
    MRR: float
    Connected_Evidence_Recall_at_5: float
    Connected_Evidence_Recall_at_10: float
    Query_Evidence_Connectivity_at_10: float
    Path_Recall_at_10: str | float
    Edge_Recall_at_10: str | float
    Retrieval_Latency_per_Query: float


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


def default_type_weights() -> dict[str, float]:
    return {
        "query_overlap": 0.8,
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
    type_weights: dict[str, float] = field(default_factory=default_type_weights)


@dataclass(frozen=True)
class RerankResult:
    ranked_nodes: list[RankedNode]
    retrieved_subgraph: RetrievedSubgraph
    score_breakdown: ScoreBreakdown | None = None


class Retriever(Protocol):
    method_name: str

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        ...
