from __future__ import annotations

from typing import Literal, TypedDict

from graph_memory.contracts.common import NodeId, TaskId


class TaskImportanceRecord(TypedDict):
    task_id: TaskId
    content_digest: str
    scores: dict[NodeId, int]


class ImportanceArtifact(TypedDict):
    schema_version: Literal[1]
    method: Literal["memory_stream"]
    tasks: list[TaskImportanceRecord]


__all__ = [
    "ImportanceArtifact",
    "TaskImportanceRecord",
]
