from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.errors import ContractValidationError
from graph_memory.contracts.tasks import MemoryItem, MemoryTaskInput
from graph_memory.retrieval.methods.memory_stream.contracts import (
    ImportanceGenerationRecord,
    ImportanceMessage,
    ImportanceSettings,
)
IMPORTANCE_PROMPT_VERSION = "memory-stream-importance-v2"

_SYSTEM_PROMPT = """You rate the long-term importance or poignancy of memory sentences.
Use an absolute 1-10 scale: 1 is routine or low-information, 10 is critical,
salient, or likely worth retaining for future reasoning. Rate each item
independently. Return exactly one score for each input item, in the same order.
Do not return node ids. Return only JSON with this shape:
{"scores":[<integer 1-10>,<integer 1-10>,...]}"""


def build_importance_messages(
    task_input: MemoryTaskInput,
    prompt_version: str = IMPORTANCE_PROMPT_VERSION,
) -> list[ImportanceMessage]:
    if prompt_version != IMPORTANCE_PROMPT_VERSION:
        raise ValueError(f"Unsupported Memory Stream importance prompt version: {prompt_version}")
    items = [
        {
            "node_id": item["id"],
            "source": item["source"],
            "text": item["text"],
            "position": item["position"],
        }
        for item in _ordered_memory_items(task_input)
    ]
    payload = {
        "items": items,
        "output_format": {"scores": ["<integer 1-10>", "<integer 1-10>", "..."]},
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _canonical_json(payload, sort_keys=True)},
    ]


def generation_record_from_settings(settings: ImportanceSettings) -> ImportanceGenerationRecord:
    return {
        "do_sample": settings.do_sample,
        "use_cache": settings.use_cache,
        "max_new_tokens": settings.max_new_tokens,
    }


def canonical_importance_payload(task_input: MemoryTaskInput, settings: ImportanceSettings) -> dict[str, object]:
    return {
        "model_id": settings.model_id,
        "prompt_version": settings.prompt_version,
        "generation": generation_record_from_settings(settings),
        "items": _semantic_items(task_input),
    }


def importance_cache_digest(task_input: MemoryTaskInput, settings: ImportanceSettings) -> str:
    return _sha256_json(canonical_importance_payload(task_input, settings))


def importance_content_digest(task_input: MemoryTaskInput) -> str:
    return _sha256_json({"items": _semantic_items(task_input)})


def parse_importance_response(response_text: str, task_input: MemoryTaskInput) -> dict[NodeId, int]:
    task_id = task_input["task_id"]
    payload = _parse_json_object(_strip_optional_json_fence(response_text), task_id=task_id)
    unknown = sorted(set(payload) - {"scores"})
    if unknown:
        raise ContractValidationError(f"Invalid importance response: task_id={task_id} unknown fields={unknown}.")
    scores = payload.get("scores")
    if not isinstance(scores, list):
        raise ContractValidationError(f"Invalid importance response: task_id={task_id} scores must be an array.")
    return _validate_scores(scores, task_input)


def task_node_ids(task_input: MemoryTaskInput) -> list[NodeId]:
    return [item["id"] for item in _ordered_memory_items(task_input)]


def _validate_scores(scores: Sequence[object], task_input: MemoryTaskInput) -> dict[NodeId, int]:
    task_id = task_input["task_id"]
    expected_ids = task_node_ids(task_input)
    if len(scores) != len(expected_ids):
        raise ContractValidationError(
            "Invalid importance response: "
            f"task_id={task_id} score count mismatch expected={len(expected_ids)} observed={len(scores)}."
        )
    validated: dict[NodeId, int] = {}
    for node_id, value in zip(expected_ids, scores, strict=True):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ContractValidationError(
                f"Invalid importance response: task_id={task_id} node_id={node_id} score must be an integer."
            )
        if value < 1 or value > 10:
            raise ContractValidationError(
                f"Invalid importance response: task_id={task_id} node_id={node_id} score must be 1-10."
            )
        validated[node_id] = value
    return validated


def _ordered_memory_items(task_input: MemoryTaskInput) -> list[MemoryItem]:
    task_id = task_input["task_id"]
    memory_items = task_input.get("memory_items")
    if not isinstance(memory_items, list) or not memory_items:
        raise ContractValidationError(f"Invalid memory stream task: task_id={task_id} memory_items must be non-empty.")
    seen_ids: set[str] = set()
    ordered: list[MemoryItem] = []
    for index, item in enumerate(memory_items):
        if not isinstance(item, dict):
            raise ContractValidationError(
                f"Invalid memory stream task: task_id={task_id} memory item index={index} is not an object."
            )
        node_id = item.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ContractValidationError(f"Invalid memory stream task: task_id={task_id} missing node id.")
        if node_id in seen_ids:
            raise ContractValidationError(f"Invalid memory stream task: task_id={task_id} duplicate node_id={node_id}.")
        seen_ids.add(node_id)
        for field_name in ("source", "text"):
            if not isinstance(item.get(field_name), str):
                raise ContractValidationError(
                    f"Invalid memory stream task: task_id={task_id} node_id={node_id} field={field_name} must be a string."
                )
        position = item.get("position")
        if not isinstance(position, int) or isinstance(position, bool):
            raise ContractValidationError(
                f"Invalid memory stream task: task_id={task_id} node_id={node_id} position must be an integer."
            )
        ordered.append(cast(MemoryItem, item))
    return ordered


def _semantic_items(task_input: MemoryTaskInput) -> list[dict[str, object]]:
    return [
        {
            "id": item["id"],
            "source": item["source"],
            "text": item["text"],
            "position": item["position"],
        }
        for item in _ordered_memory_items(task_input)
    ]


def _strip_optional_json_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        raise ContractValidationError("Invalid importance response: malformed JSON code fence.")
    opening = lines[0].strip()
    closing = lines[-1].strip()
    if opening not in {"```", "```json"} or closing != "```":
        raise ContractValidationError("Invalid importance response: malformed JSON code fence.")
    body = "\n".join(lines[1:-1]).strip()
    if body.startswith("```") or "```" in body:
        raise ContractValidationError("Invalid importance response: multiple JSON code fences.")
    return body


def _parse_json_object(payload: str, *, task_id: str) -> dict[str, object]:
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as error:
        if isinstance(error, ContractValidationError):
            raise
        raise ContractValidationError(f"Invalid importance response: task_id={task_id} JSON parse failed: {error}") from error
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid importance response: task_id={task_id} response must be an object.")
    return cast(dict[str, object], value)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise ContractValidationError(f"Invalid importance response: duplicate key={key}.")
        record[key] = value
    return record


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, object], *, sort_keys: bool) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


__all__ = [
    "IMPORTANCE_PROMPT_VERSION",
    "build_importance_messages",
    "canonical_importance_payload",
    "generation_record_from_settings",
    "importance_cache_digest",
    "importance_content_digest",
    "parse_importance_response",
    "task_node_ids",
]
