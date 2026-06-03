from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from graph_memory.contracts.common import TaskId
from graph_memory.contracts.tasks import CombinedMemoryTask, MemoryTaskInput, MemoryTaskLabels


def combined_memory_tasks(
    task_inputs: Sequence[MemoryTaskInput], task_labels: Sequence[MemoryTaskLabels]
) -> list[CombinedMemoryTask]:
    labels_by_task_id: dict[TaskId, MemoryTaskLabels] = {label["task_id"]: label for label in task_labels}
    combined: list[CombinedMemoryTask] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        matching_labels = labels_by_task_id.get(task_id)
        if matching_labels is None:
            raise ValueError(f"Cannot combine task_id={task_id}: matching labels are missing.")
        combined.append(cast(CombinedMemoryTask, cast(object, {**task_input, **matching_labels})))
    return combined

