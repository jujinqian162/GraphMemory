from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast


class MemoryStreamScoringConfigRecord(TypedDict):
    relevance_weight: float
    recency_weight: float
    importance_weight: float
    recency_decay: float


MEMORY_STREAM_SCORING_FIELDS: frozenset[str] = frozenset(
    {
        "relevance_weight",
        "recency_weight",
        "importance_weight",
        "recency_decay",
    }
)


@dataclass(frozen=True)
class MemoryStreamScoringConfig:
    relevance_weight: float = 1.0
    recency_weight: float = 0.0
    importance_weight: float = 0.01
    recency_decay: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relevance_weight",
            _non_negative_weight(
                cast(object, self.relevance_weight),
                "relevance_weight",
            ),
        )
        object.__setattr__(
            self,
            "recency_weight",
            _non_negative_weight(
                cast(object, self.recency_weight),
                "recency_weight",
            ),
        )
        object.__setattr__(
            self,
            "importance_weight",
            _non_negative_weight(
                cast(object, self.importance_weight),
                "importance_weight",
            ),
        )

        if (
            self.relevance_weight
            + self.recency_weight
            + self.importance_weight
            <= 0.0
        ):
            raise ValueError("Memory Stream scoring requires at least one positive weight.")

        raw_recency_decay = cast(object, self.recency_decay)
        if isinstance(raw_recency_decay, bool) or not isinstance(
            raw_recency_decay,
            (int, float),
        ):
            raise ValueError("Memory Stream recency_decay must be a finite number.")
        recency_decay = float(raw_recency_decay)
        if (
            not math.isfinite(recency_decay)
            or recency_decay <= 0.0
            or recency_decay > 1.0
        ):
            raise ValueError(
                "Memory Stream recency_decay must satisfy 0 < recency_decay <= 1."
            )
        object.__setattr__(self, "recency_decay", recency_decay)


def parse_memory_stream_scoring_config(
    record: Mapping[str, object],
) -> MemoryStreamScoringConfig:
    unknown_fields = sorted(set(record) - MEMORY_STREAM_SCORING_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"Memory Stream scoring config contains unsupported fields: {unknown_fields}."
        )
    missing_fields = sorted(MEMORY_STREAM_SCORING_FIELDS - set(record))
    if missing_fields:
        raise ValueError(
            f"Memory Stream scoring config is missing fields: {missing_fields}."
        )
    return MemoryStreamScoringConfig(
        relevance_weight=_number_field(record, "relevance_weight"),
        recency_weight=_number_field(record, "recency_weight"),
        importance_weight=_number_field(record, "importance_weight"),
        recency_decay=_number_field(record, "recency_decay"),
    )


def memory_stream_scoring_config_record(
    config: MemoryStreamScoringConfig,
) -> MemoryStreamScoringConfigRecord:
    return {
        "relevance_weight": config.relevance_weight,
        "recency_weight": config.recency_weight,
        "importance_weight": config.importance_weight,
        "recency_decay": config.recency_decay,
    }


def _number_field(record: Mapping[str, object], field_name: str) -> float:
    value = record[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Memory Stream scoring config field {field_name} must be a finite number."
        )
    return float(value)


def _non_negative_weight(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Memory Stream {field_name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Memory Stream {field_name} must be non-negative.")
    return number


__all__ = [
    "MEMORY_STREAM_SCORING_FIELDS",
    "MemoryStreamScoringConfig",
    "MemoryStreamScoringConfigRecord",
    "memory_stream_scoring_config_record",
    "parse_memory_stream_scoring_config",
]
