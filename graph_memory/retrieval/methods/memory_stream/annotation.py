from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.methods.memory_stream.cache import ImportanceCache
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceAnnotationResult,
    ImportanceArtifact,
    ImportanceCacheStats,
    ImportanceSettings,
    TaskImportanceRecord,
)
from graph_memory.retrieval.methods.memory_stream.prompt import (
    build_importance_messages,
    generation_record_from_settings,
    importance_content_digest,
    parse_importance_response,
)
from graph_memory.retrieval.methods.memory_stream.runtime import ImportanceRuntime
from graph_memory.validation.importance import validate_importance_artifact

RuntimeFactory = Callable[[ImportanceSettings], ImportanceRuntime]


def annotate_importance_tasks(
    task_inputs: Sequence[MemoryTaskInput],
    settings: ImportanceSettings,
    *,
    cache_dir: str | Path,
    runtime_factory: RuntimeFactory,
) -> ImportanceAnnotationResult:
    cache = ImportanceCache(cache_dir)
    records_by_task_id: dict[str, TaskImportanceRecord] = {}
    misses: list[MemoryTaskInput] = []
    hits = 0
    writes = 0

    for task_input in task_inputs:
        cached = cache.read(task_input, settings)
        if cached is None:
            misses.append(task_input)
            continue
        records_by_task_id[task_input["task_id"]] = cached
        hits += 1

    model_load_count = 0
    model_load_seconds = 0.0
    generation_calls = 0
    generated_tokens = 0
    generation_seconds = 0.0
    device: str | None = None

    if misses:
        runtime = runtime_factory(settings)
        load_stats = runtime.load()
        model_load_count = 1
        load_seconds_value = load_stats.get("model_load_seconds", 0.0)
        model_load_seconds = float(load_seconds_value) if isinstance(load_seconds_value, int | float) else 0.0
        device_value = load_stats.get("device")
        device = str(device_value) if device_value is not None else None
        for task_input in misses:
            messages = build_importance_messages(task_input, settings.prompt_version)
            generated = runtime.generate(messages, settings)
            scores = parse_importance_response(generated.text, task_input)
            task_record: TaskImportanceRecord = {
                "task_id": task_input["task_id"],
                "content_digest": importance_content_digest(task_input),
                "scores": scores,
            }
            cache.write(task_input, settings, task_record)
            writes += 1
            records_by_task_id[task_input["task_id"]] = task_record
            generation_calls += 1
            generated_tokens += generated.generated_tokens
            generation_seconds += generated.generation_seconds

    artifact: ImportanceArtifact = {
        "method": "memory_stream",
        "model": settings.model_id,
        "prompt_version": settings.prompt_version,
        "generation": generation_record_from_settings(settings),
        "tasks": [records_by_task_id[task_input["task_id"]] for task_input in task_inputs],
    }
    validate_importance_artifact(artifact, list(task_inputs))
    return ImportanceAnnotationResult(
        artifact=artifact,
        cache_stats=ImportanceCacheStats(hits=hits, misses=len(misses), writes=writes),
        model_load_count=model_load_count,
        model_load_seconds=model_load_seconds,
        generation_calls=generation_calls,
        generated_tokens=generated_tokens,
        generation_seconds=generation_seconds,
        device=device,
    )


__all__ = ["RuntimeFactory", "annotate_importance_tasks"]
