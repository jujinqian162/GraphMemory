from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.retrieval.methods.memory_stream.config import (
    MEMORY_STREAM_SCORING_FIELDS,
    MemoryStreamScoringConfig,
    parse_memory_stream_scoring_config,
)
from graph_memory.tuning.grid_search import ParameterGrid


def memory_stream_grid_from_record(
    record: Mapping[str, object],
) -> list[MemoryStreamScoringConfig]:
    unknown_fields = sorted(set(record) - MEMORY_STREAM_SCORING_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"Memory Stream grid config contains unsupported fields: {unknown_fields}."
        )
    missing_fields = sorted(MEMORY_STREAM_SCORING_FIELDS - set(record))
    if missing_fields:
        raise ValueError(
            f"Memory Stream grid config is missing fields: {missing_fields}."
        )

    parameters: dict[str, Sequence[object]] = {
        field_name: _candidate_values(record, field_name)
        for field_name in (
            "relevance_weight",
            "recency_weight",
            "importance_weight",
            "recency_decay",
        )
    }
    return [
        parse_memory_stream_scoring_config(candidate)
        for candidate in ParameterGrid(parameters=parameters, fixed={}).expand()
    ]


def _candidate_values(
    record: Mapping[str, object],
    field_name: str,
) -> list[object]:
    value = record[field_name]
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Memory Stream grid config requires a non-empty list for {field_name}."
        )
    return cast(list[object], value)


__all__ = ["memory_stream_grid_from_record"]
