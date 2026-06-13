from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

from graph_memory.contracts.common import NodeId, TaskId


class ImportanceGenerationRecord(TypedDict):
    do_sample: bool
    use_cache: bool
    max_new_tokens: int


class TaskImportanceRecord(TypedDict):
    task_id: TaskId
    content_digest: str
    scores: dict[NodeId, int]


class ImportanceArtifact(TypedDict):
    method: Literal["memory_stream"]
    model: str
    prompt_version: str
    generation: ImportanceGenerationRecord
    tasks: list[TaskImportanceRecord]


class ImportanceCacheRecord(TypedDict):
    method: Literal["memory_stream"]
    model: str
    prompt_version: str
    generation: ImportanceGenerationRecord
    cache_digest: str
    task: TaskImportanceRecord


class ImportanceMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


class ImportanceSettings(Protocol):
    @property
    def model_id(self) -> str:
        ...

    @property
    def model_path(self) -> Path:
        ...

    @property
    def prompt_version(self) -> str:
        ...

    @property
    def device(self) -> str:
        ...

    @property
    def trust_remote_code(self) -> bool:
        ...

    @property
    def torch_dtype(self) -> str:
        ...

    @property
    def low_cpu_mem_usage(self) -> bool:
        ...

    @property
    def tp_plan(self) -> None:
        ...

    @property
    def do_sample(self) -> bool:
        ...

    @property
    def use_cache(self) -> bool:
        ...

    @property
    def max_new_tokens(self) -> int:
        ...


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    generation_seconds: float


@dataclass(frozen=True)
class ImportanceCacheStats:
    hits: int
    misses: int
    writes: int


@dataclass(frozen=True)
class ImportanceAnnotationResult:
    artifact: ImportanceArtifact
    cache_stats: ImportanceCacheStats
    model_load_count: int
    model_load_seconds: float
    generation_calls: int
    generated_tokens: int
    generation_seconds: float
    device: str | None = None


__all__ = [
    "GenerationResult",
    "ImportanceAnnotationResult",
    "ImportanceArtifact",
    "ImportanceCacheRecord",
    "ImportanceCacheStats",
    "ImportanceGenerationRecord",
    "ImportanceMessage",
    "ImportanceSettings",
    "TaskImportanceRecord",
]
