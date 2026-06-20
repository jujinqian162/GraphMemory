from __future__ import annotations

from typing import Literal, TypedDict

from typing_extensions import NotRequired

from graph_memory.contracts.common import EdgeType, JsonValue, NodeId, TaskId


class QuestionNode(TypedDict):
    id: Literal["q"]
    node_type: Literal["question"]
    text: str


class GraphItemNode(TypedDict):
    id: NodeId
    node_type: Literal["graph_item"]
    node_kind: str
    text: str
    source_ref: NotRequired[str]
    group_key: NotRequired[str]
    sequence_index: NotRequired[int]
    metadata: NotRequired[dict[str, JsonValue]]


GraphNode = QuestionNode | GraphItemNode


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


__all__ = ["GraphEdge", "GraphItemNode", "GraphNode", "MemoryGraph", "QuestionNode"]
