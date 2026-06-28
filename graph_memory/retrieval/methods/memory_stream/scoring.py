from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.errors import ContractValidationError
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.retrieval.requests import TemporalMemoryRankingRequest

SECONDS_PER_DAY = 86_400.0
_WEEKDAY_MARKER = re.compile(r"\s+\([A-Za-z]{3}\)")


@dataclass(frozen=True)
class RawMemoryStreamSignals:
    relevance_by_node_id: Mapping[NodeId, float]
    recency_by_node_id: Mapping[NodeId, float]
    importance_by_node_id: Mapping[NodeId, float]


@dataclass(frozen=True)
class NormalizedMemoryStreamSignals:
    relevance_by_node_id: Mapping[NodeId, float]
    recency_by_node_id: Mapping[NodeId, float]
    importance_by_node_id: Mapping[NodeId, float]


def normalize_task_signal(raw_scores_by_node_id: Mapping[NodeId, float]) -> dict[NodeId, float]:
    """Map one task-local signal to 0..1, with constant signals mapped to 0.0."""
    if not raw_scores_by_node_id:
        return {}
    min_score = min(raw_scores_by_node_id.values())
    max_score = max(raw_scores_by_node_id.values())
    if max_score == min_score:
        return {node_id: 0.0 for node_id in raw_scores_by_node_id}
    scale = max_score - min_score
    return {
        node_id: (score - min_score) / scale
        for node_id, score in raw_scores_by_node_id.items()
    }


def normalize_memory_stream_signals(raw_signals: RawMemoryStreamSignals) -> NormalizedMemoryStreamSignals:
    """Normalize relevance, recency, and importance independently."""
    node_ids = _all_node_ids(raw_signals)
    return NormalizedMemoryStreamSignals(
        relevance_by_node_id=normalize_task_signal(_zero_fill(raw_signals.relevance_by_node_id, node_ids)),
        recency_by_node_id=normalize_task_signal(_zero_fill(raw_signals.recency_by_node_id, node_ids)),
        importance_by_node_id=normalize_task_signal(_zero_fill(raw_signals.importance_by_node_id, node_ids)),
    )


def memory_stream_recency_scores(
    request: TemporalMemoryRankingRequest,
    *,
    decay: float,
) -> dict[NodeId, float]:
    """Compute the request-owned recency signal for Memory Stream."""
    raw_mode = request.metadata.get("recency_mode")
    if raw_mode is None or raw_mode == "position":
        return pseudo_recency_scores(request, decay=decay)
    if raw_mode == "real_time":
        return real_time_recency_scores(request, decay=decay)
    if not isinstance(raw_mode, str):
        raise ContractValidationError(
            f"Invalid temporal memory request: task_id={request.task_id} recency_mode must be a string."
        )
    raise ContractValidationError(
        f"Invalid temporal memory request: task_id={request.task_id} unsupported recency_mode={raw_mode!r}."
    )


def pseudo_recency_scores(
    request: TemporalMemoryRankingRequest,
    *,
    decay: float,
) -> dict[NodeId, float]:
    """Compute decay ** (max_position - position) for each temporal candidate."""
    position_by_item_id = _position_by_item_id(request)
    if not position_by_item_id:
        return {}
    max_position = max(position_by_item_id.values())
    return {
        item_id: decay ** (max_position - position)
        for item_id, position in position_by_item_id.items()
    }


def real_time_recency_scores(
    request: TemporalMemoryRankingRequest,
    *,
    decay: float,
) -> dict[NodeId, float]:
    """Compute decay ** age_days against the latest visible temporal anchor."""
    raw_question_datetime = request.metadata.get("question_datetime")
    if not isinstance(raw_question_datetime, str) or not raw_question_datetime:
        raise ContractValidationError(
            "Invalid temporal memory request: "
            f"task_id={request.task_id} real_time recency requires question_datetime."
        )
    question_datetime = _parse_temporal_datetime(
        raw_question_datetime,
        task_id=request.task_id,
        field_name="question_datetime",
    )

    datetime_by_item_id = _datetime_by_item_id(request)
    recency_anchor = max([question_datetime, *datetime_by_item_id.values()])
    return {
        item_id: decay ** _age_days(
            recency_anchor,
            item_datetime,
            task_id=request.task_id,
            item_id=item_id,
        )
        for item_id, item_datetime in datetime_by_item_id.items()
    }


def score_memory_stream(
    normalized_signals: NormalizedMemoryStreamSignals,
    *,
    config: MemoryStreamScoringConfig,
) -> dict[NodeId, float]:
    """Return weighted final score per node id."""
    node_ids = _all_node_ids(normalized_signals)
    return {
        node_id: (
            config.relevance_weight
            * normalized_signals.relevance_by_node_id.get(node_id, 0.0)
            + config.recency_weight
            * normalized_signals.recency_by_node_id.get(node_id, 0.0)
            + config.importance_weight
            * normalized_signals.importance_by_node_id.get(node_id, 0.0)
        )
        for node_id in node_ids
    }


def rank_memory_stream_scores(score_by_node_id: Mapping[NodeId, float]) -> list[tuple[NodeId, float]]:
    """Sort by descending score and ascending node id."""
    return sorted(score_by_node_id.items(), key=lambda item: (-item[1], item[0]))


def _position_by_item_id(request: TemporalMemoryRankingRequest) -> dict[NodeId, int]:
    raw_positions = request.metadata.get("position_by_item_id")
    if not isinstance(raw_positions, Mapping):
        raise ContractValidationError(
            f"Invalid temporal memory request: task_id={request.task_id} missing position_by_item_id metadata."
        )
    expected_ids = {candidate.item_id for candidate in request.candidates}
    positions: dict[NodeId, int] = {}
    for item_id, position in raw_positions.items():
        if not isinstance(item_id, str) or not item_id:
            raise ContractValidationError(
                f"Invalid temporal memory request: task_id={request.task_id} position item ids must be non-empty strings."
            )
        if not isinstance(position, int) or isinstance(position, bool):
            raise ContractValidationError(
                f"Invalid temporal memory request: task_id={request.task_id} item_id={item_id} position must be an integer."
            )
        positions[item_id] = position
    missing = sorted(expected_ids - set(positions))
    extra = sorted(set(positions) - expected_ids)
    if missing or extra:
        raise ContractValidationError(
            f"Invalid temporal memory request: task_id={request.task_id} position ids mismatch missing={missing} extra={extra}."
        )
    return positions


def _datetime_by_item_id(request: TemporalMemoryRankingRequest) -> dict[NodeId, datetime]:
    raw_datetimes = request.metadata.get("datetime_by_item_id")
    if not isinstance(raw_datetimes, Mapping):
        raise ContractValidationError(
            "Invalid temporal memory request: "
            f"task_id={request.task_id} real_time recency requires datetime_by_item_id metadata."
        )
    expected_ids = {candidate.item_id for candidate in request.candidates}
    datetimes: dict[NodeId, datetime] = {}
    for item_id, raw_datetime in raw_datetimes.items():
        if not isinstance(item_id, str) or not item_id:
            raise ContractValidationError(
                f"Invalid temporal memory request: task_id={request.task_id} datetime item ids must be non-empty strings."
            )
        if not isinstance(raw_datetime, str) or not raw_datetime:
            raise ContractValidationError(
                "Invalid temporal memory request: "
                f"task_id={request.task_id} item_id={item_id} datetime must be a non-empty string."
            )
        datetimes[item_id] = _parse_temporal_datetime(
            raw_datetime,
            task_id=request.task_id,
            field_name=f"datetime_by_item_id[{item_id}]",
        )
    missing = sorted(expected_ids - set(datetimes))
    extra = sorted(set(datetimes) - expected_ids)
    if missing or extra:
        raise ContractValidationError(
            f"Invalid temporal memory request: task_id={request.task_id} datetime ids mismatch missing={missing} extra={extra}."
        )
    return datetimes


def _parse_temporal_datetime(value: str, *, task_id: str, field_name: str) -> datetime:
    stripped = value.strip()
    iso_value = stripped.replace("Z", "+00:00")
    try:
        return _as_naive_utc(datetime.fromisoformat(iso_value))
    except ValueError:
        pass

    without_weekday = _WEEKDAY_MARKER.sub("", stripped)
    for pattern in (
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(without_weekday, pattern)
        except ValueError:
            continue
    raise ContractValidationError(
        "Invalid temporal memory request: "
        f"task_id={task_id} {field_name} has unsupported datetime format: {value!r}."
    )


def _as_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _age_days(
    recency_anchor: datetime,
    item_datetime: datetime,
    *,
    task_id: str,
    item_id: str,
) -> float:
    age_seconds = (recency_anchor - item_datetime).total_seconds()
    if age_seconds < 0.0:
        raise ContractValidationError(
            "Invalid temporal memory request: "
            f"task_id={task_id} item_id={item_id} datetime is after recency anchor."
        )
    return age_seconds / SECONDS_PER_DAY


def _all_node_ids(signals: RawMemoryStreamSignals | NormalizedMemoryStreamSignals) -> set[NodeId]:
    return set(signals.relevance_by_node_id) | set(signals.recency_by_node_id) | set(signals.importance_by_node_id)


def _zero_fill(scores_by_node_id: Mapping[NodeId, float], node_ids: set[NodeId]) -> dict[NodeId, float]:
    return {node_id: scores_by_node_id.get(node_id, 0.0) for node_id in node_ids}


__all__ = [
    "NormalizedMemoryStreamSignals",
    "RawMemoryStreamSignals",
    "memory_stream_recency_scores",
    "normalize_memory_stream_signals",
    "normalize_task_signal",
    "pseudo_recency_scores",
    "rank_memory_stream_scores",
    "real_time_recency_scores",
    "score_memory_stream",
]
