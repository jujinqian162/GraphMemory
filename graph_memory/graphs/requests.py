from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import JsonValue, TaskId

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True)
class GraphBuildNode:
    node_id: str
    text: str
    node_kind: str
    source_ref: str | None
    group_key: str | None
    sequence_index: int | None
    metadata: Mapping[str, JsonScalar]


@dataclass(frozen=True)
class GraphBuildEdge:
    source: str
    target: str
    edge_type: str
    weight: float
    directed: bool
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True)
class GraphBuildRequest:
    task_id: TaskId
    query_text: str
    nodes: Sequence[GraphBuildNode]
    input_visible_edges: Sequence[GraphBuildEdge]


__all__ = ["GraphBuildEdge", "GraphBuildNode", "GraphBuildRequest"]
