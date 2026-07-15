from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from graph_memory.datasets.traject_bench.records import (
    TrajectBenchExample,
    TrajectBenchPartition,
    TrajectBenchToolDefinition,
    TrajectBenchToolParameter,
    TrajectBenchTrajectoryType,
)

QUERY_FIELDS = {
    "query",
    "tool list",
    "tool_list",
    "trajectory_type",
    "tool count",
    "final_answer",
    "domain",
    "executable",
    "generation_info",
    "num_successful_tools",
    "num_tools_used",
    "sequence_description",
    "sequence_name",
    "task_description",
    "task_name",
    "tool_count",
}
QUERY_TOOL_FIELDS = {
    "tool name",
    "tool description",
    "required parameters",
    "optional parameters",
    "executed_output",
    "execution_status",
    "API name",
    "domain name",
    "parent tool name",
    "sequence_step",
    "original_description",
    "adapt_parameter",
    "adapt_constraint",
    "adapt_reason",
}
CATALOG_TOOL_FIELDS = {
    "API name",
    "category",
    "code",
    "connected tools",
    "domain name",
    "optional_parameters",
    "output_info",
    "parent tool description",
    "parent tool name",
    "required_parameters",
    "tool description",
    "tool name",
}
CATALOG_PARAMETER_FIELDS = {"default", "description", "name", "type"}
CONNECTED_TOOL_FIELDS = {"connect params", "tool name"}


def resolve_traject_bench_data_root(source: Path) -> Path:
    source = Path(source).resolve()
    candidates = (source, source / "public_data")
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("parallel", "sequential", "tools")):
            return candidate
    raise ValueError(
        "TRAJECT-Bench source must contain parallel, sequential, and tools "
        f"directories: {source}"
    )


def partition_for_split(split: str) -> TrajectBenchPartition:
    if split == "train":
        return "parallel_simple"
    if split == "dev":
        return "parallel_hard"
    if split == "test":
        return "sequential"
    raise ValueError(f"Unsupported TRAJECT-Bench workflow split: {split!r}.")


def discover_traject_bench_query_files(source: Path, split: str) -> tuple[Path, ...]:
    root = resolve_traject_bench_data_root(source)
    partition = partition_for_split(split)
    if partition == "parallel_simple":
        files = list((root / "parallel").glob("*/simple_ver.json"))
    elif partition == "parallel_hard":
        files = list((root / "parallel").glob("*/hard_ver.json"))
    else:
        files = list((root / "sequential").glob("*/traj_query.json"))
        travel_simple = root / "sequential" / "Travel" / "simple_ver.json"
        if travel_simple.is_file():
            files.append(travel_simple)
    ordered = tuple(
        sorted(
            files,
            key=lambda path: (
                path.relative_to(root).as_posix().casefold(),
                path.relative_to(root).as_posix(),
            ),
        )
    )
    if not ordered:
        raise ValueError(
            f"TRAJECT-Bench split={split} discovered no query files under {root}."
        )
    return ordered


def parse_traject_bench_query(
    raw_record: object,
    *,
    source_key: str,
    source_index: int,
    partition: TrajectBenchPartition,
    domain_name: str,
) -> TrajectBenchExample:
    path = f"TRAJECT-Bench query {source_key}:{source_index}"
    record = _required_record(raw_record, path)
    _reject_unknown_fields(record, QUERY_FIELDS, path)
    query = _required_string(record, "query", path)
    raw_tool_list = _query_tool_list(record, path)
    tool_names = tuple(
        _parse_query_tool_name(raw_tool, path=f"{path} tool[{index}]")
        for index, raw_tool in enumerate(raw_tool_list)
    )
    if not tool_names:
        raise ValueError(f"{path} tool list must be non-empty.")
    _validate_declared_tool_count(record, len(tool_names), path)
    trajectory_type = _trajectory_type(record, partition=partition, path=path)
    declared_domain = record.get("domain")
    if declared_domain is not None and declared_domain != domain_name:
        raise ValueError(
            f"{path} domain={declared_domain!r} does not match source domain={domain_name!r}."
        )
    return TrajectBenchExample(
        source_key=source_key,
        source_index=source_index,
        partition=partition,
        domain_name=domain_name,
        query=query,
        trajectory_type=trajectory_type,
        tool_names=tool_names,
        final_answer=_final_answer(record.get("final_answer"), path),
    )


def parse_traject_bench_tool_catalog(
    raw_records: object,
    *,
    domain_name: str,
) -> list[TrajectBenchToolDefinition]:
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(
            f"TRAJECT-Bench domain={domain_name} tool catalog must be a non-empty list."
        )
    return [
        _parse_catalog_tool(
            raw_record,
            domain_name=domain_name,
            catalog_index=index,
        )
        for index, raw_record in enumerate(raw_records)
    ]


def _parse_catalog_tool(
    raw_record: object,
    *,
    domain_name: str,
    catalog_index: int,
) -> TrajectBenchToolDefinition:
    path = f"TRAJECT-Bench domain={domain_name} catalog[{catalog_index}]"
    record = _required_record(raw_record, path)
    _reject_unknown_fields(record, CATALOG_TOOL_FIELDS, path)
    _required_string(record, "domain name", path)
    parameters = (
        *_parse_catalog_parameters(record, "required_parameters", path, required=True),
        *_parse_catalog_parameters(record, "optional_parameters", path, required=False),
    )
    connected_tool_names = _parse_connected_tool_names(record, path)
    return TrajectBenchToolDefinition(
        tool_name=_required_string(record, "tool name", path),
        tool_description=_optional_string(record, "tool description", path),
        parent_tool_name=_required_string(record, "parent tool name", path),
        parent_tool_description=_optional_string(
            record,
            "parent tool description",
            path,
        ),
        api_name=_required_string(record, "API name", path),
        domain_name=domain_name,
        category=_optional_string(record, "category", path),
        parameters=parameters,
        connected_tool_names=connected_tool_names,
    )


def _parse_catalog_parameters(
    record: Mapping[str, object],
    field_name: str,
    path: str,
    *,
    required: bool,
) -> tuple[TrajectBenchToolParameter, ...]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"{path} field={field_name} must be a list.")
    parameters: list[TrajectBenchToolParameter] = []
    for index, raw_parameter in enumerate(value):
        parameter_path = f"{path} {field_name}[{index}]"
        parameter = _required_record(raw_parameter, parameter_path)
        _reject_unknown_fields(parameter, CATALOG_PARAMETER_FIELDS, parameter_path)
        parameters.append(
            TrajectBenchToolParameter(
                name=_required_string(parameter, "name", parameter_path),
                parameter_type=_optional_string(parameter, "type", parameter_path),
                description=_optional_string(
                    parameter,
                    "description",
                    parameter_path,
                ),
                required=required,
            )
        )
    return tuple(parameters)


def _parse_connected_tool_names(
    record: Mapping[str, object],
    path: str,
) -> tuple[str, ...]:
    value = record.get("connected tools", [])
    if not isinstance(value, list):
        raise ValueError(f"{path} field=connected tools must be a list.")
    names: list[str] = []
    for index, raw_connection in enumerate(value):
        connection_path = f"{path} connected tools[{index}]"
        connection = _required_record(raw_connection, connection_path)
        _reject_unknown_fields(connection, CONNECTED_TOOL_FIELDS, connection_path)
        names.append(_required_string(connection, "tool name", connection_path))
    return tuple(names)


def _query_tool_list(
    record: Mapping[str, object],
    path: str,
) -> Sequence[object]:
    fields = [field_name for field_name in ("tool list", "tool_list") if field_name in record]
    if len(fields) != 1:
        raise ValueError(
            f"{path} must contain exactly one of 'tool list' or 'tool_list'."
        )
    field_name = fields[0]
    value = record[field_name]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path} field={field_name} must contain a JSON tool array."
            ) from error
    if not isinstance(value, list):
        raise ValueError(f"{path} field={field_name} must be a list or JSON array text.")
    return cast(Sequence[object], value)


def _parse_query_tool_name(raw_tool: object, *, path: str) -> str:
    record = _required_record(raw_tool, path)
    _reject_unknown_fields(record, QUERY_TOOL_FIELDS, path)
    return _required_string(record, "tool name", path)


def _trajectory_type(
    record: Mapping[str, object],
    *,
    partition: TrajectBenchPartition,
    path: str,
) -> TrajectBenchTrajectoryType:
    expected: TrajectBenchTrajectoryType = (
        "sequential" if partition == "sequential" else "parallel"
    )
    value = record.get("trajectory_type", expected)
    if value != expected:
        raise ValueError(
            f"{path} trajectory_type={value!r} does not match partition={partition}."
        )
    return expected


def _validate_declared_tool_count(
    record: Mapping[str, object],
    actual_count: int,
    path: str,
) -> None:
    for field_name in ("tool count", "tool_count", "num_tools_used"):
        value = record.get(field_name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} field={field_name} must be an integer.")
        if value != actual_count:
            raise ValueError(
                f"{path} field={field_name} value={value} does not match "
                f"tool list size={actual_count}."
            )


def _final_answer(value: object, path: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping):
        answer = value.get("answer")
        if isinstance(answer, str) and answer:
            return answer
    raise ValueError(f"{path} final_answer must be text or an object with answer text.")


def _required_record(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(
    record: Mapping[str, object],
    field_name: str,
    path: str,
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} field={field_name} must be non-empty text.")
    return value


def _optional_string(
    record: Mapping[str, object],
    field_name: str,
    path: str,
) -> str:
    value = record.get(field_name, "")
    if not isinstance(value, str):
        raise ValueError(f"{path} field={field_name} must be text.")
    return value


def _reject_unknown_fields(
    record: Mapping[str, object],
    allowed_fields: set[str],
    path: str,
) -> None:
    unknown = sorted(set(record) - allowed_fields)
    if unknown:
        raise ValueError(f"{path} unknown fields={unknown}.")


__all__ = [
    "discover_traject_bench_query_files",
    "parse_traject_bench_query",
    "parse_traject_bench_tool_catalog",
    "partition_for_split",
    "resolve_traject_bench_data_root",
]
