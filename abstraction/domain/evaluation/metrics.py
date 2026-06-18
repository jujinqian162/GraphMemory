from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from abstraction.domain.common.identifiers import MetricSuiteId, TaskId


@dataclass(frozen=True)
class MetricSelectionKey:
    metric_suite_id: MetricSuiteId
    prediction_kind: str
    eval_unit_kind: str


@dataclass(frozen=True)
class MetricRow:
    task_id: TaskId
    metric_name: str
    metric_value: float
    metric_group: str


@dataclass(frozen=True)
class MetricResultTable:
    metric_suite_id: MetricSuiteId
    rows: Sequence[MetricRow]
    aggregate_values: Mapping[str, float]

