from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from graph_memory.contracts.common import NodeId, TaskId, TrainPairSampleType


class TrainPairRecord(TypedDict):
    task_id: TaskId
    node_id: NodeId
    label: Literal[0, 1]
    sample_type: TrainPairSampleType


class TrainPairBuildSummary(TypedDict):
    positive_count: int
    negative_count_by_type: dict[str, int]
    avg_positive_per_task: float
    avg_negative_per_task: float
    tasks_with_no_positive: list[TaskId]
    sampling_config: dict[str, object]
    requested_negative_count_by_type: NotRequired[dict[str, int]]
    shortfall_by_type: NotRequired[dict[str, int]]
    overlap_count_by_type: NotRequired[dict[str, int]]
    source_overlap_by_task: NotRequired[dict[str, dict[str, list[str]]]]


__all__ = ["TrainPairBuildSummary", "TrainPairRecord"]
