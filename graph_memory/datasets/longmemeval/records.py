from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from graph_memory.contracts.common import JsonValue


class LongMemEvalTurnItem(TypedDict):
    item_id: str
    session_id: str
    session_order: int
    turn_index: int
    global_position: int
    role: str
    datetime: str
    text: str


class LongMemEvalRankingRecord(TypedDict):
    task_id: str
    question: str
    question_datetime: str
    candidate_items: list[LongMemEvalTurnItem]
    metadata: dict[str, JsonValue]


class LongMemEvalLabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_support_item_ids: list[str]
    gold_support_session_ids: list[str]
    gold_dependency_edges: list[list[str]]
    metadata: dict[str, JsonValue]


class CombinedLongMemEvalRecord(LongMemEvalRankingRecord, LongMemEvalLabelRecord):
    """Combined LongMemEval inspection artifact; retrieval code must not consume it."""


@dataclass(frozen=True)
class LongMemEvalTurn:
    role: str
    content: str
    has_answer: bool


@dataclass(frozen=True)
class LongMemEvalSession:
    session_id: str
    datetime: str
    turns: tuple[LongMemEvalTurn, ...]


@dataclass(frozen=True)
class LongMemEvalExample:
    raw_id: str
    question_type: str
    question: str
    answer: str
    question_datetime: str
    sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConvertedLongMemEvalExample:
    ranking_record: LongMemEvalRankingRecord
    label_record: LongMemEvalLabelRecord


@dataclass(frozen=True)
class LongMemEvalConversionResult:
    ranking_records: list[LongMemEvalRankingRecord]
    label_records: list[LongMemEvalLabelRecord]
