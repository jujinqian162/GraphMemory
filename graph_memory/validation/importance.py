from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.artifact import (
    importance_content_digest,
    task_node_ids,
)
from graph_memory.contracts.errors import ContractValidationError

ARTIFACT_FIELDS = {"schema_version", "method", "tasks"}
TASK_RECORD_FIELDS = {"task_id", "content_digest", "scores"}


def validate_importance_artifact(artifact: object, task_inputs: Sequence[MemoryTaskInput]) -> None:
    tasks = _validate_artifact_envelope(artifact)
    if len(tasks) != len(task_inputs):
        raise ContractValidationError(
            f"Invalid importance artifact: task count mismatch expected={len(task_inputs)} observed={len(tasks)}."
        )
    for index, (task_record, task_input) in enumerate(zip(tasks, task_inputs, strict=True)):
        if not isinstance(task_record, Mapping):
            raise ContractValidationError(f"Invalid importance artifact: task record index={index} is not an object.")
        record = cast(Mapping[str, object], task_record)
        if record.get("task_id") != task_input["task_id"]:
            raise ContractValidationError(
                f"Invalid importance artifact: task order mismatch index={index} expected={task_input['task_id']} observed={record.get('task_id')}."
            )
        validate_task_importance_record(record, task_input)


def select_importance_records(
    artifact: object,
    task_inputs: Sequence[MemoryTaskInput],
) -> list[TaskImportanceRecord]:
    task_records = _validate_artifact_envelope(artifact)
    records_by_task_id: dict[str, Mapping[str, object]] = {}
    for index, task_record in enumerate(task_records):
        if not isinstance(task_record, Mapping):
            raise ContractValidationError(
                f"Invalid importance artifact: task record index={index} is not an object."
            )
        record = cast(Mapping[str, object], task_record)
        _reject_unknown(record, TASK_RECORD_FIELDS, "task importance record")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ContractValidationError(
                f"Invalid importance artifact: task record index={index} task_id must be a non-empty string."
            )
        if task_id in records_by_task_id:
            raise ContractValidationError(
                f"Invalid importance artifact: duplicate task_id={task_id}."
            )
        records_by_task_id[task_id] = record

    selected: list[TaskImportanceRecord] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        task_record = records_by_task_id.get(task_id)
        if task_record is None:
            raise ContractValidationError(
                f"Invalid importance artifact: missing task_id={task_id}."
            )
        validate_task_importance_record(task_record, task_input)
        selected.append(cast(TaskImportanceRecord, task_record))
    return selected


def validate_task_importance_record(record: object, task_input: MemoryTaskInput) -> None:
    task_record = _require_mapping(record, "task importance record")
    task_id = task_input["task_id"]
    _reject_unknown(task_record, TASK_RECORD_FIELDS, "task importance record", task_id=task_id)
    if task_record.get("task_id") != task_id:
        raise ContractValidationError(
            f"Invalid task importance record: task_id={task_id} observed task_id={task_record.get('task_id')}."
        )
    expected_digest = importance_content_digest(task_input)
    if task_record.get("content_digest") != expected_digest:
        raise ContractValidationError(
            f"Invalid task importance record: task_id={task_id} content_digest mismatch."
        )
    scores = task_record.get("scores")
    if not isinstance(scores, Mapping):
        raise ContractValidationError(f"Invalid task importance record: task_id={task_id} scores must be an object.")
    _validate_score_mapping(cast(Mapping[str, object], scores), task_input, artifact_name="task importance record")


def _validate_artifact_envelope(artifact: object) -> list[object]:
    record = _require_mapping(artifact, "importance artifact")
    _reject_unknown(record, ARTIFACT_FIELDS, "importance artifact")
    if record.get("schema_version") != 1:
        raise ContractValidationError("Invalid importance artifact: schema_version must be 1.")
    if record.get("method") != "memory_stream":
        raise ContractValidationError("Invalid importance artifact: method must be memory_stream.")
    tasks = record.get("tasks")
    if not isinstance(tasks, list):
        raise ContractValidationError("Invalid importance artifact: tasks must be a list.")
    return cast(list[object], tasks)


def _validate_score_mapping(
    scores: Mapping[str, object],
    task_input: MemoryTaskInput,
    *,
    artifact_name: str,
) -> None:
    task_id = task_input["task_id"]
    expected_ids = set(task_node_ids(task_input))
    observed_ids = set(scores)
    if expected_ids != observed_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} missing={missing} extra={extra}.")
    for node_id in task_node_ids(task_input):
        value = scores[node_id]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ContractValidationError(
                f"Invalid {artifact_name}: task_id={task_id} node_id={node_id} score must be an integer."
            )
        if value < 1 or value > 10:
            raise ContractValidationError(
                f"Invalid {artifact_name}: task_id={task_id} node_id={node_id} score must be 1-10."
            )


def _require_mapping(value: object, artifact_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be an object.")
    return cast(Mapping[str, object], value)


def _reject_unknown(
    record: Mapping[str, object],
    allowed: set[str],
    artifact_name: str,
    *,
    task_id: str | None = None,
) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} unknown fields={unknown}.")


__all__ = [
    "select_importance_records",
    "validate_importance_artifact",
    "validate_task_importance_record",
]
