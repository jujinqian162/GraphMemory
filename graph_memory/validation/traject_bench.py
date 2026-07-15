from __future__ import annotations

from graph_memory.datasets.traject_bench.converter import (
    stable_traject_bench_tool_id,
)
from graph_memory.validation.common import (
    ContractValidationError,
    _reject_unknown_fields,
    _require_record_list,
    _require_record_map,
    _require_unique,
    _required_int,
    _required_string,
    validate_no_label_fields,
    validate_task_id_alignment,
)

TRAJECT_BENCH_RANKING_FIELDS = {
    "task_id",
    "query",
    "trajectory_type",
    "domain_name",
    "candidate_tools",
    "metadata",
}
TRAJECT_BENCH_CANDIDATE_FIELDS = {
    "tool_id",
    "tool_name",
    "text",
    "parent_tool_name",
    "api_name",
    "domain_name",
    "catalog_position",
    "connected_tool_ids",
}
TRAJECT_BENCH_LABEL_FIELDS = {
    "task_id",
    "gold_answer",
    "gold_tool_ids",
    "gold_tool_sequence_ids",
    "gold_dependency_edges",
    "metadata",
}


def validate_traject_bench_ranking_records(records: object) -> None:
    values = _require_record_list(records, "TRAJECT-Bench ranking records")
    seen_task_ids: set[str] = set()
    for record_index, record in enumerate(values):
        task_id = _required_string(record, "task_id", "TRAJECT-Bench ranking record")
        _reject_unknown_fields(
            record,
            TRAJECT_BENCH_RANKING_FIELDS,
            "TRAJECT-Bench ranking record",
            task_id,
        )
        validate_no_label_fields(
            record,
            artifact_name="TRAJECT-Bench ranking record",
            task_id=task_id,
        )
        _require_unique(task_id, seen_task_ids, "TRAJECT-Bench ranking task_id")
        if not task_id.startswith("traject_bench_"):
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench ranking record: task_id={task_id} "
                "must start with traject_bench_."
            )
        _required_string(record, "query", "TRAJECT-Bench ranking record", task_id)
        trajectory_type = _required_string(
            record,
            "trajectory_type",
            "TRAJECT-Bench ranking record",
            task_id,
        )
        if trajectory_type not in {"parallel", "sequential"}:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench ranking record: task_id={task_id} "
                f"trajectory_type={trajectory_type!r}."
            )
        domain_name = _required_string(
            record,
            "domain_name",
            "TRAJECT-Bench ranking record",
            task_id,
        )
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("dataset") != "traject_bench":
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench ranking record: task_id={task_id} metadata "
                "must identify dataset=traject_bench."
            )
        candidates = record.get("candidate_tools")
        if not isinstance(candidates, list) or not candidates:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench ranking record: task_id={task_id} "
                "candidate_tools must be a non-empty list."
            )
        _validate_candidates(
            candidates,
            task_id=task_id,
            domain_name=domain_name,
            record_index=record_index,
        )


def validate_traject_bench_label_records(
    labels: object,
    records_by_task_id: object,
) -> None:
    values = _require_record_list(labels, "TRAJECT-Bench label records")
    ranking_by_task_id = _require_record_map(
        records_by_task_id,
        "TRAJECT-Bench ranking records by task_id",
    )
    seen_task_ids: set[str] = set()
    for label in values:
        task_id = _required_string(label, "task_id", "TRAJECT-Bench label record")
        _reject_unknown_fields(
            label,
            TRAJECT_BENCH_LABEL_FIELDS,
            "TRAJECT-Bench label record",
            task_id,
        )
        _require_unique(task_id, seen_task_ids, "TRAJECT-Bench label task_id")
        ranking = ranking_by_task_id.get(task_id)
        if ranking is None:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} has no "
                "matching ranking record."
            )
        _required_string(label, "gold_answer", "TRAJECT-Bench label record", task_id)
        metadata = label.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("dataset") != "traject_bench":
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} metadata "
                "must identify dataset=traject_bench."
            )
        candidate_ids = _candidate_ids(ranking, task_id=task_id)
        gold_tool_ids = _id_list(
            label.get("gold_tool_ids"),
            field_name="gold_tool_ids",
            task_id=task_id,
            allow_duplicates=False,
        )
        sequence_ids = _id_list(
            label.get("gold_tool_sequence_ids"),
            field_name="gold_tool_sequence_ids",
            task_id=task_id,
            allow_duplicates=True,
        )
        if any(tool_id not in candidate_ids for tool_id in (*gold_tool_ids, *sequence_ids)):
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} gold tool "
                "IDs must exist in candidate_tools."
            )
        if list(dict.fromkeys(sequence_ids)) != gold_tool_ids:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} "
                "gold_tool_ids must equal first occurrences from gold_tool_sequence_ids."
            )
        trajectory_type = ranking.get("trajectory_type")
        expected_edges = _expected_dependency_edges(
            sequence_ids,
            sequential=trajectory_type == "sequential",
        )
        actual_edges = _dependency_edges(
            label.get("gold_dependency_edges"),
            task_id=task_id,
            valid_ids=set(gold_tool_ids),
        )
        if actual_edges != expected_edges:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} dependency "
                f"edges={actual_edges} expected={expected_edges}."
            )
    validate_task_id_alignment(
        "TRAJECT-Bench ranking/label join",
        set(ranking_by_task_id),
        seen_task_ids,
    )


def _validate_candidates(
    candidates: list[object],
    *,
    task_id: str,
    domain_name: str,
    record_index: int,
) -> None:
    candidate_records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench ranking record: task_id={task_id} "
                f"candidate index={position} is not an object."
            )
        _reject_unknown_fields(
            candidate,
            TRAJECT_BENCH_CANDIDATE_FIELDS,
            "TRAJECT-Bench candidate tool",
            task_id,
        )
        tool_id = _required_string(
            candidate,
            "tool_id",
            "TRAJECT-Bench candidate tool",
            task_id,
        )
        tool_name = _required_string(
            candidate,
            "tool_name",
            "TRAJECT-Bench candidate tool",
            task_id,
        )
        _require_unique(tool_id, seen_ids, f"TRAJECT-Bench candidate ID task_id={task_id}")
        _require_unique(
            tool_name,
            seen_names,
            f"TRAJECT-Bench candidate name task_id={task_id}",
        )
        if tool_id != stable_traject_bench_tool_id(tool_name):
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench candidate: task_id={task_id} "
                f"tool_name={tool_name!r} has unstable tool_id={tool_id}."
            )
        _required_string(candidate, "text", "TRAJECT-Bench candidate tool", task_id)
        _required_string(
            candidate,
            "parent_tool_name",
            "TRAJECT-Bench candidate tool",
            task_id,
        )
        _required_string(
            candidate,
            "api_name",
            "TRAJECT-Bench candidate tool",
            task_id,
        )
        candidate_domain = _required_string(
            candidate,
            "domain_name",
            "TRAJECT-Bench candidate tool",
            task_id,
        )
        if candidate_domain != domain_name:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench candidate: task_id={task_id} domain "
                f"{candidate_domain!r} does not match record domain={domain_name!r}."
            )
        catalog_position = _required_int(
            candidate,
            "catalog_position",
            "TRAJECT-Bench candidate tool",
            task_id,
            minimum=0,
        )
        if catalog_position != position:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench candidate: task_id={task_id} "
                f"catalog_position={catalog_position} expected={position} "
                f"record_index={record_index}."
            )
        candidate_records.append(candidate)

    for candidate in candidate_records:
        tool_id = str(candidate["tool_id"])
        connected_ids = _id_list(
            candidate.get("connected_tool_ids"),
            field_name="connected_tool_ids",
            task_id=task_id,
            allow_duplicates=False,
            allow_empty=True,
        )
        if tool_id in connected_ids or any(target not in seen_ids for target in connected_ids):
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench candidate: task_id={task_id} connected tool "
                "IDs must be distinct existing non-self candidates."
            )


def _candidate_ids(ranking: dict[str, object], *, task_id: str) -> set[str]:
    candidates = ranking.get("candidate_tools")
    if not isinstance(candidates, list):
        raise ContractValidationError(
            f"Invalid TRAJECT-Bench ranking record: task_id={task_id} candidate_tools missing."
        )
    return {
        candidate["tool_id"]
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("tool_id"), str)
    }


def _id_list(
    value: object,
    *,
    field_name: str,
    task_id: str,
    allow_duplicates: bool,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "a list" if allow_empty else "a non-empty list"
        raise ContractValidationError(
            f"Invalid TRAJECT-Bench record: task_id={task_id} "
            f"field={field_name} must be {requirement}."
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench record: task_id={task_id} "
                f"field={field_name} entries must be non-empty strings."
            )
        result.append(item)
    if not allow_duplicates and len(result) != len(set(result)):
        raise ContractValidationError(
            f"Invalid TRAJECT-Bench record: task_id={task_id} "
            f"field={field_name} contains duplicates."
        )
    return result


def _dependency_edges(
    value: object,
    *,
    task_id: str,
    valid_ids: set[str],
) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise ContractValidationError(
            f"Invalid TRAJECT-Bench label record: task_id={task_id} "
            "gold_dependency_edges must be a list."
        )
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(value):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} "
                f"gold_dependency_edges[{index}] must be [source, target]."
            )
        source, target = edge
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or source not in valid_ids
            or target not in valid_ids
            or source == target
        ):
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} edge "
                f"{edge!r} must contain distinct gold tool IDs."
            )
        pair = (source, target)
        if pair in seen:
            raise ContractValidationError(
                f"Invalid TRAJECT-Bench label record: task_id={task_id} duplicate "
                f"dependency edge={pair}."
            )
        seen.add(pair)
        result.append(pair)
    return result


def _expected_dependency_edges(
    sequence_ids: list[str],
    *,
    sequential: bool,
) -> list[tuple[str, str]]:
    if not sequential:
        return []
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in zip(sequence_ids, sequence_ids[1:]):
        edge = (source, target)
        if source == target or edge in seen:
            continue
        seen.add(edge)
        result.append(edge)
    return result


__all__ = [
    "validate_traject_bench_label_records",
    "validate_traject_bench_ranking_records",
]
