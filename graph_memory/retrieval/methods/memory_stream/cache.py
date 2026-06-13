from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.infrastructure.io import read_json, write_json_atomic
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceCacheRecord,
    ImportanceSettings,
    TaskImportanceRecord,
)
from graph_memory.retrieval.methods.memory_stream.prompt import (
    generation_record_from_settings,
    importance_cache_digest,
)
from graph_memory.validation.importance import validate_importance_cache_record


class ImportanceCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for_digest(self, digest: str) -> Path:
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def path_for_task(self, task_input: MemoryTaskInput, settings: ImportanceSettings) -> Path:
        return self.path_for_digest(importance_cache_digest(task_input, settings))

    def read(self, task_input: MemoryTaskInput, settings: ImportanceSettings) -> TaskImportanceRecord | None:
        digest = importance_cache_digest(task_input, settings)
        path = self.path_for_digest(digest)
        try:
            record = read_json(path)
        except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        try:
            validate_importance_cache_record(
                record,
                task_input,
                model_id=settings.model_id,
                prompt_version=settings.prompt_version,
                generation=generation_record_from_settings(settings),
                cache_digest=digest,
            )
        except ValueError:
            return None
        return cast(ImportanceCacheRecord, record)["task"]

    def write(
        self,
        task_input: MemoryTaskInput,
        settings: ImportanceSettings,
        task_record: TaskImportanceRecord,
    ) -> Path:
        digest = importance_cache_digest(task_input, settings)
        record: ImportanceCacheRecord = {
            "method": "memory_stream",
            "model": settings.model_id,
            "prompt_version": settings.prompt_version,
            "generation": generation_record_from_settings(settings),
            "cache_digest": digest,
            "task": task_record,
        }
        path = self.path_for_digest(digest)
        write_json_atomic(path, record)
        return path


__all__ = ["ImportanceCache"]
