from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

from abstraction.domain.common.identifiers import ItemId, TaskId


@dataclass(frozen=True)
class EvidenceEvalUnit:
    task_id: TaskId
    predicted_item_ids: Sequence[ItemId]
    gold_item_ids: Sequence[ItemId]


@dataclass(frozen=True)
class SupportCoverageEvalUnit:
    task_id: TaskId
    predicted_item_ids: Sequence[ItemId]
    gold_support_item_ids: Sequence[ItemId]
    item_to_turn_id: Mapping[ItemId, str]
    item_to_session_id: Mapping[ItemId, str]


@dataclass(frozen=True)
class MultiHopEvalUnit:
    task_id: TaskId
    predicted_item_ids: Sequence[ItemId]
    gold_paragraph_item_ids: Sequence[ItemId]
    gold_entity_ids: Sequence[str]
    gold_relation_path_labels: Sequence[str]


@dataclass(frozen=True)
class AnswerEvalUnit:
    task_id: TaskId
    predicted_answer_text: str
    gold_answer_text: str
    citation_item_ids: Sequence[ItemId]


EvaluationUnit: TypeAlias = EvidenceEvalUnit | SupportCoverageEvalUnit | MultiHopEvalUnit | AnswerEvalUnit
EvaluationUnitBatch: TypeAlias = Sequence[EvaluationUnit]

