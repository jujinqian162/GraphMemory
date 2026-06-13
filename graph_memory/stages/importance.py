from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.infrastructure.io import read_json, write_json_atomic
from graph_memory.registry.stage_configs import ImportanceStageConfig
from graph_memory.retrieval.methods.memory_stream.annotation import annotate_importance_tasks
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceAnnotationResult
from graph_memory.retrieval.methods.memory_stream.runtime import LocalTransformersImportanceRuntime
from graph_memory.validation import validate_memory_task_inputs


@dataclass(frozen=True)
class ImportanceStageResult:
    annotation: ImportanceAnnotationResult


def run_importance_stage(
    config: ImportanceStageConfig,
    *,
    task_inputs: Sequence[MemoryTaskInput] | None = None,
) -> ImportanceStageResult:
    if task_inputs is None:
        loaded = read_json(config.io.tasks)
        validate_memory_task_inputs(loaded)
        task_list = list(loaded)
    else:
        task_list = list(task_inputs)
        validate_memory_task_inputs(task_list)
    result = annotate_importance_tasks(
        task_list,
        config.job,
        cache_dir=config.io.cache_dir,
        runtime_factory=lambda settings: LocalTransformersImportanceRuntime(settings),
    )
    write_json_atomic(config.io.output, result.artifact)
    return ImportanceStageResult(annotation=result)


__all__ = ["ImportanceStageResult", "run_importance_stage"]
