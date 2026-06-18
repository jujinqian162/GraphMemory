from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

from abstraction.domain.common.capability_names import RequestKind
from abstraction.domain.common.identifiers import ItemId, RequestId, TaskId


@dataclass(frozen=True)
class TextRankingRequest:
    request_id: RequestId
    request_kind: RequestKind
    task_id: TaskId
    query_text: str
    candidate_text_by_item: Mapping[ItemId, str]


@dataclass(frozen=True)
class GraphRankingRequest:
    request_id: RequestId
    request_kind: RequestKind
    task_id: TaskId
    query_ref: str
    candidate_item_ids: Sequence[ItemId]
    graph_ref: str
    seed_scores_by_item: Mapping[ItemId, float]


@dataclass(frozen=True)
class TemporalMemoryRankingRequest:
    request_id: RequestId
    request_kind: RequestKind
    task_id: TaskId
    query_text: str
    memory_item_ids: Sequence[ItemId]
    memory_text_by_item: Mapping[ItemId, str]
    recency_signal_by_item: Mapping[ItemId, float]
    importance_signal_by_item: Mapping[ItemId, float]


@dataclass(frozen=True)
class ContextGatheringRequest:
    request_id: RequestId
    request_kind: RequestKind
    task_id: TaskId
    question_text: str
    text_store_ref: str
    graph_or_session_context_ref: str | None
    candidate_context_item_ids: Sequence[ItemId]
    candidate_text_by_item: Mapping[ItemId, str]


@dataclass(frozen=True)
class AnswerRequest:
    request_id: RequestId
    request_kind: RequestKind
    task_id: TaskId
    question_text: str
    retrieved_context_refs: Sequence[str]
    asset_refs: Sequence[str]


RetrievalRequest: TypeAlias = (
    TextRankingRequest
    | GraphRankingRequest
    | TemporalMemoryRankingRequest
    | ContextGatheringRequest
    | AnswerRequest
)
