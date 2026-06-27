from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from graph_memory.contracts.metrics import MetricRow

MetricSelectionKey: TypeAlias = Callable[[MetricRow], tuple[float, ...]]


def retrieval_tuning_objective(row: MetricRow) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )


def retrieval_candidate_key(
    row: MetricRow,
) -> tuple[float, float, float, float]:
    return (
        retrieval_tuning_objective(row),
        float(row.get("Full Support@10", 0.0)),
        -float(row.get("Retrieval Latency / Query", 0.0)),
        -float(row.get("Avg Retrieved Edges", 0.0)),
    )


def longmemeval_retrieval_candidate_key(
    row: MetricRow,
) -> tuple[float, float, float, float, float]:
    return (
        _required_float(row, "Full Turn Support@10"),
        _required_float(row, "Turn Recall@10"),
        _required_float(row, "Session Recall@10"),
        float(row["MRR"]),
        -float(row.get("Retrieval Latency / Query", 0.0)),
    )


def _required_float(row: MetricRow, column: str) -> float:
    value = row.get(column)
    if value is None:
        raise KeyError(f"Metric row is missing required tuning column: {column}")
    return float(value)


__all__ = [
    "MetricSelectionKey",
    "longmemeval_retrieval_candidate_key",
    "retrieval_candidate_key",
    "retrieval_tuning_objective",
]
