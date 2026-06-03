from __future__ import annotations

from typing import NotRequired, TypedDict

from graph_memory.contracts.common import MethodName, NodeId, TaskId
from graph_memory.contracts.graphs import GraphEdge


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


__all__ = ["RankedNodeRecord", "RankedResult", "RetrievedSubgraph"]
