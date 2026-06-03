from __future__ import annotations

from typing import Literal, TypedDict

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


__all__ = ["TrainPairBuildSummary", "TrainPairRecord"]
