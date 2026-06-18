from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from abstraction.domain.common.capability_names import ViewKind
from abstraction.domain.common.identifiers import DatasetId, ItemId, TaskId, ViewId


class TaskView(Protocol):
    view_id: ViewId
    view_kind: ViewKind
    dataset_id: DatasetId
    task_id: TaskId


@dataclass(frozen=True)
class CandidateTextItem:
    item_id: ItemId
    text_ref: str
    visible_metadata: Mapping[str, str]


@dataclass(frozen=True)
class EvidenceRankingView:
    view_id: ViewId
    view_kind: ViewKind
    dataset_id: DatasetId
    task_id: TaskId
    query_text: str
    candidate_items: Sequence[CandidateTextItem]


@dataclass(frozen=True)
class ContextGatheringView:
    view_id: ViewId
    view_kind: ViewKind
    dataset_id: DatasetId
    task_id: TaskId
    question_text: str
    context_items: Sequence[CandidateTextItem]
    context_groups: Mapping[str, Sequence[ItemId]]
    asset_refs: Sequence[str]


@dataclass(frozen=True)
class GraphBuildNodeView:
    item_id: ItemId
    node_text_ref: str
    grouping_metadata: Mapping[str, str]
    visible_structural_hints: Mapping[str, str]


@dataclass(frozen=True)
class GraphBuildView:
    view_id: ViewId
    view_kind: ViewKind
    dataset_id: DatasetId
    task_id: TaskId
    query_text: str
    candidate_nodes: Sequence[GraphBuildNodeView]
    input_visible_edges: Sequence[tuple[ItemId, ItemId, str]]
    asset_refs: Sequence[str]


@dataclass(frozen=True)
class TrainingView:
    view_id: ViewId
    view_kind: ViewKind
    dataset_id: DatasetId
    task_id: TaskId
    training_query_ref: str
    positive_item_ids: Sequence[ItemId]
    negative_item_ids: Sequence[ItemId]
    optional_graph_context_ref: str | None


@dataclass(frozen=True)
class AnswerEvaluationView:
    view_id: ViewId
    view_kind: ViewKind
    dataset_id: DatasetId
    task_id: TaskId
    question_text: str
    answer_context_refs: Sequence[str]

