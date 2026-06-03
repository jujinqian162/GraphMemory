from __future__ import annotations

from graph_memory.validation.common import (
    ContractValidationError,
    _memory_node_ids,
    _reject_unknown_fields,
    _require_record_list,
    _require_record_map,
    _require_unique,
    _required_int,
    _required_string,
    validate_no_label_fields,
)

MEMORY_TASK_INPUT_FIELDS = {"task_id", "query", "memory_items", "metadata", "debug"}
MEMORY_ITEM_FIELDS = {"id", "node_type", "text", "source", "sentence_id", "position"}
LABEL_FIELDS = {"task_id", "gold_answer", "gold_evidence_nodes", "gold_dependency_edges", "metadata", "debug"}


def validate_memory_task_inputs(records: object) -> None:
    records = _require_record_list(records, "memory task inputs")

    seen_task_ids: set[str] = set()
    for index, task_input in enumerate(records):
        if not isinstance(task_input, dict):
            raise ContractValidationError(f"Invalid memory task inputs: record index={index} is not an object.")
        task_id = _required_string(task_input, "task_id", "memory task input")
        _reject_unknown_fields(task_input, MEMORY_TASK_INPUT_FIELDS, "memory task input", task_id)
        validate_no_label_fields(task_input, artifact_name="memory task input", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "memory task input task_id")
        _required_string(task_input, "query", "memory task input", task_id)

        memory_items = task_input.get("memory_items")
        if not isinstance(memory_items, list) or not memory_items:
            raise ContractValidationError(f"Invalid memory task input: task_id={task_id} memory_items must be non-empty.")

        seen_node_ids: set[str] = set()
        for expected_position, memory_item in enumerate(memory_items):
            if not isinstance(memory_item, dict):
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} memory item index={expected_position} is not an object."
                )
            _reject_unknown_fields(memory_item, MEMORY_ITEM_FIELDS, "memory item", task_id)
            node_id = _required_string(memory_item, "id", "memory item", task_id)
            _require_unique(node_id, seen_node_ids, f"memory item id task_id={task_id}")
            if node_id != f"m{expected_position}":
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} expected id=m{expected_position}."
                )
            if memory_item.get("node_type") != "document_sentence":
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} node_type must be document_sentence."
                )
            _required_string(memory_item, "text", "memory item", task_id)
            _required_string(memory_item, "source", "memory item", task_id)
            _required_int(memory_item, "sentence_id", "memory item", task_id, minimum=0)
            position = _required_int(memory_item, "position", "memory item", task_id, minimum=0)
            if position != expected_position:
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} position={position} expected {expected_position}."
                )


def validate_memory_task_labels(records: object, inputs_by_task_id: object) -> None:
    records = _require_record_list(records, "memory task labels")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")

    seen_task_ids: set[str] = set()
    for index, task_labels in enumerate(records):
        if not isinstance(task_labels, dict):
            raise ContractValidationError(f"Invalid memory task labels: record index={index} is not an object.")
        task_id = _required_string(task_labels, "task_id", "memory task labels")
        _reject_unknown_fields(task_labels, LABEL_FIELDS, "memory task labels", task_id)
        _require_unique(task_id, seen_task_ids, "memory task labels task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid memory task labels: task_id={task_id} has no matching input task.")

        _required_string(task_labels, "gold_answer", "memory task labels", task_id)
        gold_nodes = task_labels.get("gold_evidence_nodes")
        if not isinstance(gold_nodes, list) or not gold_nodes:
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_evidence_nodes must be a non-empty list."
            )
        if len(gold_nodes) != len(set(gold_nodes)):
            raise ContractValidationError(f"Invalid memory task labels: task_id={task_id} duplicate gold evidence node.")

        valid_node_ids = _memory_node_ids(inputs_by_task_id[task_id])
        for node_id in gold_nodes:
            if not isinstance(node_id, str) or node_id not in valid_node_ids:
                raise ContractValidationError(
                    f"Invalid memory task labels: task_id={task_id} gold node={node_id} does not exist in input task."
                )

        dependency_edges = task_labels.get("gold_dependency_edges")
        if not isinstance(dependency_edges, list):
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_dependency_edges must be a list."
            )
        if dependency_edges:
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_dependency_edges must be empty for HotpotQA Phase 1."
            )


__all__ = ["validate_memory_task_inputs", "validate_memory_task_labels", "validate_no_label_fields"]
