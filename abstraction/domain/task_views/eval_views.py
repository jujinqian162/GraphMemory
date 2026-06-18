from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from abstraction.domain.common.identifiers import DatasetId, ItemId, TaskId, ViewId


class EvalLabelView(Protocol):
    view_id: ViewId
    dataset_id: DatasetId
    task_id: TaskId
    label_visibility: str


@dataclass(frozen=True)
class EvidenceEvalView:
    view_id: ViewId
    dataset_id: DatasetId
    task_id: TaskId
    label_visibility: str
    gold_evidence_item_ids: Sequence[ItemId]
    optional_dependency_labels: Mapping[str, Sequence[ItemId]]


@dataclass(frozen=True)
class LongMemEvalEvalView:
    view_id: ViewId
    dataset_id: DatasetId
    task_id: TaskId
    label_visibility: str
    gold_answer_text: str
    gold_support_item_ids: Sequence[ItemId]
    item_to_turn_id: Mapping[ItemId, str]
    item_to_session_id: Mapping[ItemId, str]
    task_metadata: Mapping[str, str]


@dataclass(frozen=True)
class MultiHopEvalView:
    view_id: ViewId
    dataset_id: DatasetId
    task_id: TaskId
    label_visibility: str
    gold_paragraph_item_ids: Sequence[ItemId]
    gold_entity_ids: Sequence[str]
    gold_relation_path_labels: Sequence[str]
    gold_answer_text: str

