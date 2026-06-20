from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.contracts.errors import ContractValidationError
from graph_memory.datasets.hotpotqa.records import CombinedHotpotQARecord, HotpotQALabelRecord, HotpotQARankingRecord


def coerce_hotpotqa_ranking_records(records: object) -> list[HotpotQARankingRecord]:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid HotpotQA canonical ranking records: artifact must be a list.")
    return [_coerce_hotpotqa_ranking_record(record) for record in records]


def coerce_hotpotqa_label_records(records: object) -> list[HotpotQALabelRecord]:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid HotpotQA canonical label records: artifact must be a list.")
    return [_coerce_hotpotqa_label_record(record) for record in records]


def combined_hotpotqa_records(
    ranking_records: Sequence[HotpotQARankingRecord],
    label_records: Sequence[HotpotQALabelRecord],
) -> list[CombinedHotpotQARecord]:
    labels_by_task_id = {record["task_id"]: record for record in label_records}
    combined: list[CombinedHotpotQARecord] = []
    for record in ranking_records:
        task_id = record["task_id"]
        matching_label = labels_by_task_id.get(task_id)
        if matching_label is None:
            raise ValueError(f"Cannot combine task_id={task_id}: matching labels are missing.")
        combined.append(cast(CombinedHotpotQARecord, cast(object, {**record, **matching_label})))
    return combined


def _coerce_hotpotqa_ranking_record(record: object) -> HotpotQARankingRecord:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Invalid HotpotQA canonical ranking record: record is not an object.")
    if "candidate_sentences" in record:
        return cast(HotpotQARankingRecord, cast(object, dict(record)))
    if "query" not in record or "memory_items" not in record:
        return cast(HotpotQARankingRecord, cast(object, dict(record)))

    task_id = _required_string(record, "task_id", "HotpotQA legacy ranking record")
    memory_items = record.get("memory_items")
    if not isinstance(memory_items, list):
        raise ContractValidationError(
            f"Invalid HotpotQA legacy ranking record: task_id={task_id} memory_items must be a list."
        )

    coerced: dict[str, object] = {
        "task_id": task_id,
        "question": _required_string(record, "query", "HotpotQA legacy ranking record", task_id),
        "candidate_sentences": [
            _coerce_candidate_sentence(memory_item, task_id=task_id)
            for memory_item in memory_items
        ],
    }
    _copy_optional(record, coerced, "metadata")
    _copy_optional(record, coerced, "debug")
    return cast(HotpotQARankingRecord, cast(object, coerced))


def _coerce_hotpotqa_label_record(record: object) -> HotpotQALabelRecord:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Invalid HotpotQA canonical label record: record is not an object.")
    if "gold_evidence_sentence_ids" in record:
        return cast(HotpotQALabelRecord, cast(object, dict(record)))
    if "gold_evidence_nodes" not in record:
        return cast(HotpotQALabelRecord, cast(object, dict(record)))

    task_id = _required_string(record, "task_id", "HotpotQA legacy label record")
    gold_evidence_nodes = record.get("gold_evidence_nodes")
    if not isinstance(gold_evidence_nodes, list):
        raise ContractValidationError(
            f"Invalid HotpotQA legacy label record: task_id={task_id} gold_evidence_nodes must be a list."
        )
    coerced: dict[str, object] = {
        "task_id": task_id,
        "gold_answer": _required_string(record, "gold_answer", "HotpotQA legacy label record", task_id),
        "gold_evidence_sentence_ids": gold_evidence_nodes,
        "gold_dependency_edges": record.get("gold_dependency_edges", []),
    }
    _copy_optional(record, coerced, "metadata")
    _copy_optional(record, coerced, "debug")
    return cast(HotpotQALabelRecord, cast(object, coerced))


def _coerce_candidate_sentence(record: object, *, task_id: str) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ContractValidationError(f"Invalid HotpotQA legacy ranking record: task_id={task_id} memory item is not an object.")
    if record.get("node_type") != "document_sentence":
        node_id = record.get("id")
        raise ContractValidationError(
            f"Invalid HotpotQA legacy ranking record: task_id={task_id} node_id={node_id} node_type must be document_sentence."
        )
    return {
        "sentence_id": _required_string(record, "id", "HotpotQA legacy memory item", task_id),
        "title": _required_string(record, "source", "HotpotQA legacy memory item", task_id),
        "sentence_index": _required_int(record, "sentence_id", "HotpotQA legacy memory item", task_id),
        "position": _required_int(record, "position", "HotpotQA legacy memory item", task_id),
        "text": _required_string(record, "text", "HotpotQA legacy memory item", task_id),
    }


def _copy_optional(source: Mapping[str, object], target: dict[str, object], field_name: str) -> None:
    if field_name in source:
        target[field_name] = source[field_name]


def _required_string(
    record: Mapping[str, object],
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be a non-empty string.")
    return value


def _required_int(record: Mapping[str, object], field_name: str, artifact_name: str, task_id: str) -> int:
    value = record.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} field={field_name} must be an integer.")
    return value
