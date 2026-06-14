from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import cast

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.errors import ContractValidationError
from graph_memory.contracts.tasks import MemoryItem, MemoryTaskInput


def importance_content_digest(task_input: MemoryTaskInput) -> str:
    payload = {"items": [_semantic_item(item) for item in _ordered_memory_items(task_input)]}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_node_ids(task_input: MemoryTaskInput) -> list[NodeId]:
    return [item["id"] for item in _ordered_memory_items(task_input)]


def _ordered_memory_items(task_input: MemoryTaskInput) -> list[MemoryItem]:
    task_id = task_input["task_id"]
    task_record = cast(Mapping[str, object], task_input)
    raw_memory_items = task_record.get("memory_items")
    if not isinstance(raw_memory_items, list) or not raw_memory_items:
        raise ContractValidationError(
            f"Invalid memory stream task: task_id={task_id} memory_items must be non-empty."
        )
    memory_items = cast(list[object], raw_memory_items)
    seen_ids: set[str] = set()
    ordered: list[MemoryItem] = []
    for index, item in enumerate(memory_items):
        if not isinstance(item, dict):
            raise ContractValidationError(
                f"Invalid memory stream task: task_id={task_id} memory item index={index} is not an object."
            )
        item_record = cast(dict[str, object], item)
        node_id = item_record.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ContractValidationError(
                f"Invalid memory stream task: task_id={task_id} missing node id."
            )
        if node_id in seen_ids:
            raise ContractValidationError(
                f"Invalid memory stream task: task_id={task_id} duplicate node_id={node_id}."
            )
        seen_ids.add(node_id)
        for field_name in ("source", "text"):
            if not isinstance(item_record.get(field_name), str):
                raise ContractValidationError(
                    f"Invalid memory stream task: task_id={task_id} node_id={node_id} field={field_name} must be a string."
                )
        position = item_record.get("position")
        if not isinstance(position, int) or isinstance(position, bool):
            raise ContractValidationError(
                f"Invalid memory stream task: task_id={task_id} node_id={node_id} position must be an integer."
            )
        ordered.append(cast(MemoryItem, cast(object, item_record)))
    return ordered


def _semantic_item(item: MemoryItem) -> dict[str, object]:
    return {
        "id": item["id"],
        "source": item["source"],
        "text": item["text"],
        "position": item["position"],
    }


__all__ = ["importance_content_digest", "task_node_ids"]
