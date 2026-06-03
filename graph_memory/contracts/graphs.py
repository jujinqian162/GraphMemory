from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from graph_memory.contracts.common import EdgeType, TaskId
from graph_memory.contracts.tasks import MemoryItem


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


__all__ = ["GraphEdge", "GraphMemoryNode", "GraphNode", "MemoryGraph", "QuestionNode"]
