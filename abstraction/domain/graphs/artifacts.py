from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from abstraction.domain.common.identifiers import ArtifactId, ItemId, TaskId


@dataclass(frozen=True)
class GraphNode:
    item_id: ItemId
    node_kind: str
    visible_metadata: Mapping[str, str]


@dataclass(frozen=True)
class GraphEdge:
    source_item_id: ItemId
    target_item_id: ItemId
    edge_kind: str
    edge_visibility: str


@dataclass(frozen=True)
class GraphArtifact:
    artifact_id: ArtifactId
    task_id: TaskId
    nodes: Sequence[GraphNode]
    edges: Sequence[GraphEdge]
    graph_metadata: Mapping[str, str]


@dataclass(frozen=True)
class GraphIndexView:
    graph_ref: str
    task_id: TaskId
    candidate_item_ids: Sequence[ItemId]
    edge_kinds: Sequence[str]

