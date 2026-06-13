from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, TypeAlias, cast

from graph_memory.contracts.errors import ContractValidationError


ValidationRecord: TypeAlias = dict[str, Any]
ValidationRecords: TypeAlias = list[ValidationRecord]
ValidationRecordMap: TypeAlias = dict[str, ValidationRecord]

FORBIDDEN_LABEL_FIELDS: set[str] = {
    "gold_answer",
    "gold_evidence_nodes",
    "gold_dependency_edges",
    "supporting_facts",
    "is_gold",
    "is_gold_evidence",
    "is_gold_edge",
}


def validate_no_label_fields(value: Any, *, artifact_name: str = "artifact", task_id: str | None = None) -> None:
    _walk_forbidden_fields(value, artifact_name=artifact_name, task_id=task_id, path=artifact_name)


def validate_task_id_alignment(artifact_name: str, expected_task_ids: set[str], observed_task_ids: set[str]) -> None:
    if expected_task_ids != observed_task_ids:
        missing = sorted(expected_task_ids - observed_task_ids)
        extra = sorted(observed_task_ids - expected_task_ids)
        raise ContractValidationError(
            f"Invalid {artifact_name}: task_id alignment mismatch; missing={missing} extra={extra}."
        )


def _walk_forbidden_fields(value: Any, *, artifact_name: str, task_id: str | None, path: str) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in FORBIDDEN_LABEL_FIELDS:
                location = f" task_id={task_id}" if task_id is not None else ""
                raise ContractValidationError(
                    f"Invalid {artifact_name}:{location} forbidden label field {key} at {path}.{key}."
                )
            _walk_forbidden_fields(nested_value, artifact_name=artifact_name, task_id=task_id, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _walk_forbidden_fields(nested_value, artifact_name=artifact_name, task_id=task_id, path=f"{path}[{index}]")


def _require_record_list(value: object, artifact_name: str) -> ValidationRecords:
    if not isinstance(value, list):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be a list.")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ContractValidationError(f"Invalid {artifact_name}: record index={index} is not an object.")
    return cast(ValidationRecords, value)


def _require_record_map(value: object, artifact_name: str) -> ValidationRecordMap:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be an object.")
    for key, record in value.items():
        if not isinstance(key, str) or not key:
            raise ContractValidationError(f"Invalid {artifact_name}: keys must be non-empty strings.")
        if not isinstance(record, dict):
            raise ContractValidationError(f"Invalid {artifact_name}: value for key={key} is not an object.")
    return cast(ValidationRecordMap, value)


def _require_record(value: object, artifact_name: str) -> ValidationRecord:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be an object.")
    return cast(ValidationRecord, value)


def _reject_unknown_fields(
    record: ValidationRecord,
    allowed_fields: set[str],
    artifact_name: str,
    task_id: str | None = None,
) -> None:
    unknown = sorted(set(record) - allowed_fields)
    if unknown:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} unknown fields={unknown}.")


def _required_string(record: ValidationRecord, field_name: str, artifact_name: str, task_id: str | None = None) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be a non-empty string.")
    return value


def _required_int(
    record: ValidationRecord,
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
    *,
    minimum: int | None = None,
) -> int:
    value = record.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be an integer.")
    if minimum is not None and value < minimum:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be >= {minimum}.")
    return value


def _required_finite_number(
    record: ValidationRecord,
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
    *,
    minimum: float | None = None,
) -> float:
    value = record.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be finite.")
    number = float(value)
    if minimum is not None and number < minimum:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be >= {minimum}.")
    return number


def _require_unique(value: str, seen_values: set[str], artifact_name: str) -> None:
    if value in seen_values:
        raise ContractValidationError(f"Invalid {artifact_name}: duplicate value={value}.")
    seen_values.add(value)


def _memory_node_ids(task_input: ValidationRecord) -> set[str]:
    memory_items = task_input.get("memory_items")
    if not isinstance(memory_items, list):
        return set()
    return {memory_item["id"] for memory_item in memory_items if isinstance(memory_item, dict) and "id" in memory_item}


def _graph_node_ids(graph: ValidationRecord) -> set[str]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return set()
    return {node["id"] for node in nodes if isinstance(node, dict) and "id" in node}


def _to_plain_dict(config: object) -> ValidationRecord:
    if isinstance(config, dict):
        return cast(ValidationRecord, config)
    if is_dataclass(config) and not isinstance(config, type):
        return cast(ValidationRecord, asdict(config))
    raise ContractValidationError("Invalid config: expected dict or dataclass instance.")


def _validate_string_sequence(value: object, field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"Invalid trainable model config: {field_name} must be a list or tuple.")
    if not allow_empty and not value:
        raise ContractValidationError(f"Invalid trainable model config: {field_name} must be non-empty.")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ContractValidationError(f"Invalid trainable model config: {field_name} entries must be non-empty strings.")
        strings.append(item)
    if len(strings) != len(set(strings)):
        raise ContractValidationError(f"Invalid trainable model config: {field_name} contains duplicate entries.")
    return tuple(strings)


def _required_attr(value: object, field_name: str, artifact_name: str) -> Any:
    if not hasattr(value, field_name):
        raise ContractValidationError(f"Invalid {artifact_name}: missing field={field_name}.")
    return getattr(value, field_name)


def _require_tensor_1d(value: object, field_name: str, artifact_name: str) -> Any:
    tensor = _required_attr(value, field_name, artifact_name)
    if getattr(tensor, "ndim", None) != 1:
        raise ContractValidationError(f"Invalid {artifact_name}: {field_name} must be a 1D tensor.")
    return tensor


def _require_tensor_2d(value: object, field_name: str, artifact_name: str) -> Any:
    tensor = _required_attr(value, field_name, artifact_name)
    if getattr(tensor, "ndim", None) != 2:
        raise ContractValidationError(f"Invalid {artifact_name}: {field_name} must be a 2D tensor.")
    return tensor
