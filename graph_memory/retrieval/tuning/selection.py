from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

MetricSelectionRow: TypeAlias = Mapping[str, object]
MetricSelectionKey: TypeAlias = Callable[[MetricSelectionRow], tuple[float, ...]]


def retrieval_tuning_objective(row: MetricSelectionRow) -> float:
    return (
        0.50 * _required_float(row, "Full Support@5")
        + 0.30 * _required_float(row, "Recall@5")
        + 0.20 * _required_float(row, "Connected Evidence Recall@10")
    )


def retrieval_candidate_key(
    row: MetricSelectionRow,
) -> tuple[float, float, float, float]:
    return (
        retrieval_tuning_objective(row),
        _optional_float(row, "Full Support@10"),
        -_optional_float(row, "Retrieval Latency / Query"),
        -_optional_float(row, "Avg Retrieved Edges"),
    )


def longmemeval_retrieval_candidate_key(
    row: MetricSelectionRow,
) -> tuple[float, float, float, float, float]:
    return (
        _required_float(row, "Full Turn Support@10"),
        _required_float(row, "Turn Recall@10"),
        _required_float(row, "Session Recall@10"),
        _required_float(row, "MRR"),
        -_optional_float(row, "Retrieval Latency / Query"),
    )


def _optional_float(row: MetricSelectionRow, column: str, default: float = 0.0) -> float:
    return _coerce_float(row.get(column, default), column)


def _required_float(row: MetricSelectionRow, column: str) -> float:
    value = row.get(column)
    if value is None:
        raise KeyError(f"Metric row is missing required tuning column: {column}")
    return _coerce_float(value, column)


def _coerce_float(value: object, column: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"Metric row column must be numeric: {column}")
    return float(value)


__all__ = [
    "MetricSelectionKey",
    "MetricSelectionRow",
    "longmemeval_retrieval_candidate_key",
    "retrieval_candidate_key",
    "retrieval_tuning_objective",
]
