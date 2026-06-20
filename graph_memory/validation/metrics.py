from __future__ import annotations

import math
from typing import Protocol

from graph_memory.validation.common import ContractValidationError, _require_record_list


class MetricRowValidator(Protocol):
    def validate_metric_rows(self, rows: object) -> None:
        ...


METRIC_COLUMNS = [
    "Method",
    "Recall@2",
    "Recall@5",
    "Recall@10",
    "Evidence F1@5",
    "Evidence F1@10",
    "Full Support@5",
    "Full Support@10",
    "MRR",
    "Connected Evidence Recall@5",
    "Connected Evidence Recall@10",
    "Query-Evidence Connectivity@10",
    "Path Recall@10",
    "Edge Recall@10",
    "Retrieval Latency / Query",
]


def validate_metric_rows(rows: object, *, metric_suite: MetricRowValidator | None = None) -> None:
    if metric_suite is not None:
        metric_suite.validate_metric_rows(rows)
        return
    validate_evidence_metric_rows(rows)


def validate_evidence_metric_rows(rows: object) -> None:
    rows = _require_record_list(rows, "metric rows")
    for row in rows:
        if not isinstance(row, dict):
            raise ContractValidationError("Invalid metric rows: row is not an object.")
        missing = [column for column in METRIC_COLUMNS if column not in row]
        if missing:
            raise ContractValidationError(f"Invalid metric rows: missing columns={missing}.")
        for column in METRIC_COLUMNS:
            if column in {"Method", "Path Recall@10", "Edge Recall@10"}:
                continue
            value = float(row[column])
            if not math.isfinite(value):
                raise ContractValidationError(f"Invalid metric rows: column={column} must be finite.")
            if column == "Retrieval Latency / Query":
                if value < 0.0:
                    raise ContractValidationError("Invalid metric rows: latency must be non-negative.")
            elif value < 0.0 or value > 1.0:
                raise ContractValidationError(f"Invalid metric rows: column={column} must be in [0.0, 1.0].")


__all__ = ["METRIC_COLUMNS", "MetricRowValidator", "validate_evidence_metric_rows", "validate_metric_rows"]
