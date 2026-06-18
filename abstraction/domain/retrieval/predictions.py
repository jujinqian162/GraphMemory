from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeAlias

from abstraction.domain.common.capability_names import PredictionKind
from abstraction.domain.common.identifiers import ItemId, PredictionId, TaskId


@dataclass(frozen=True)
class RetrievalTraceRef:
    trace_kind: str
    trace_ref: str
    visible_to_evaluation: bool


@dataclass(frozen=True)
class RankingPrediction:
    prediction_id: PredictionId
    prediction_kind: PredictionKind
    task_id: TaskId
    ranked_item_ids: Sequence[ItemId]
    score_by_item: Mapping[ItemId, float]
    latency_ms: float | None
    trace_refs: Sequence[RetrievalTraceRef]


@dataclass(frozen=True)
class ContextPrediction:
    prediction_id: PredictionId
    prediction_kind: PredictionKind
    task_id: TaskId
    retrieved_context_item_ids: Sequence[ItemId]
    context_score_by_item: Mapping[ItemId, float]
    latency_ms: float | None
    reasoning_path_refs: Sequence[RetrievalTraceRef]


@dataclass(frozen=True)
class AnswerPrediction:
    prediction_id: PredictionId
    prediction_kind: PredictionKind
    task_id: TaskId
    answer_text: str
    citation_item_ids: Sequence[ItemId]
    latency_ms: float | None


RetrievalPrediction: TypeAlias = RankingPrediction | ContextPrediction | AnswerPrediction
