from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.datasets.traject_bench.records import (
    CanonicalTrajectBenchToolCatalog,
    ConvertedTrajectBenchExample,
    PreparedTrajectBenchToolCatalog,
    TrajectBenchCandidateTool,
    TrajectBenchConversionResult,
    TrajectBenchExample,
    TrajectBenchLabelRecord,
    TrajectBenchRankingRecord,
    TrajectBenchToolDefinition,
    TrajectBenchToolParameter,
)


class MissingCatalogToolsError(ValueError):
    def __init__(self, *, source_key: str, source_index: int, tool_names: Sequence[str]) -> None:
        self.source_key = source_key
        self.source_index = source_index
        self.tool_names = tuple(tool_names)
        super().__init__(
            "TRAJECT-Bench query "
            f"{source_key}:{source_index} gold tools missing from public catalog: "
            f"{list(self.tool_names)}"
        )


def canonicalize_traject_bench_tool_catalog(
    definitions: Sequence[TrajectBenchToolDefinition],
    *,
    domain_name: str,
) -> CanonicalTrajectBenchToolCatalog:
    if not definitions:
        raise ValueError(
            f"TRAJECT-Bench domain={domain_name} tool catalog must be non-empty."
        )
    definitions_by_name: dict[str, list[TrajectBenchToolDefinition]] = defaultdict(list)
    for definition in definitions:
        if definition.domain_name != domain_name:
            raise ValueError(
                f"TRAJECT-Bench tool={definition.tool_name!r} domain "
                f"{definition.domain_name!r} does not match catalog domain={domain_name!r}."
            )
        definitions_by_name[definition.tool_name].append(definition)

    tools = tuple(
        _merge_tool_definitions(definitions_by_name[tool_name])
        for tool_name in sorted(definitions_by_name, key=_text_sort_key)
    )
    known_names = {tool.tool_name for tool in tools}
    unresolved_connections = sum(
        1
        for tool in tools
        for connected_name in tool.connected_tool_names
        if connected_name not in known_names
    )
    return CanonicalTrajectBenchToolCatalog(
        domain_name=domain_name,
        tools=tools,
        duplicate_entry_count=len(definitions) - len(tools),
        unresolved_connection_count=unresolved_connections,
    )


def prepare_traject_bench_tool_catalog(
    catalog: CanonicalTrajectBenchToolCatalog,
) -> PreparedTrajectBenchToolCatalog:
    tool_ids_by_name: dict[str, str] = {}
    names_by_tool_id: dict[str, str] = {}
    for tool in catalog.tools:
        tool_id = stable_traject_bench_tool_id(tool.tool_name)
        collision = names_by_tool_id.get(tool_id)
        if collision is not None and collision != tool.tool_name:
            raise ValueError(
                f"TRAJECT-Bench tool ID collision: {collision!r} and "
                f"{tool.tool_name!r} both map to {tool_id}."
            )
        names_by_tool_id[tool_id] = tool.tool_name
        tool_ids_by_name[tool.tool_name] = tool_id

    candidates = tuple(
        TrajectBenchCandidateTool(
            tool_id=tool_ids_by_name[tool.tool_name],
            tool_name=tool.tool_name,
            text=_candidate_text(tool),
            parent_tool_name=tool.parent_tool_name,
            api_name=tool.api_name,
            domain_name=tool.domain_name,
            catalog_position=position,
            connected_tool_ids=[
                tool_ids_by_name[connected_name]
                for connected_name in tool.connected_tool_names
                if connected_name in tool_ids_by_name
                and connected_name != tool.tool_name
            ],
        )
        for position, tool in enumerate(catalog.tools)
    )
    return PreparedTrajectBenchToolCatalog(
        domain_name=catalog.domain_name,
        candidates=candidates,
        tool_ids_by_name=tool_ids_by_name,
        duplicate_entry_count=catalog.duplicate_entry_count,
        unresolved_connection_count=catalog.unresolved_connection_count,
    )


def convert_traject_bench_example(
    example: TrajectBenchExample,
    catalog: PreparedTrajectBenchToolCatalog,
) -> ConvertedTrajectBenchExample:
    if catalog.domain_name != example.domain_name:
        raise ValueError(
            f"TRAJECT-Bench query domain={example.domain_name!r} does not match "
            f"catalog domain={catalog.domain_name!r}."
        )
    missing_names = sorted(
        {name for name in example.tool_names if name not in catalog.tool_ids_by_name},
        key=_text_sort_key,
    )
    if missing_names:
        raise MissingCatalogToolsError(
            source_key=example.source_key,
            source_index=example.source_index,
            tool_names=missing_names,
        )

    sequence_ids = [catalog.tool_ids_by_name[name] for name in example.tool_names]
    distinct_tool_ids = list(dict.fromkeys(sequence_ids))
    task_id: TaskId = _task_id(example)
    ranking_record: TrajectBenchRankingRecord = {
        "task_id": task_id,
        "query": example.query,
        "trajectory_type": example.trajectory_type,
        "domain_name": example.domain_name,
        "candidate_tools": list(catalog.candidates),
        "metadata": {
            "dataset": "traject_bench",
            "partition": example.partition,
            "candidate_pool": "domain_catalog",
            "source_key": example.source_key,
            "source_index": example.source_index,
        },
    }
    label_record: TrajectBenchLabelRecord = {
        "task_id": task_id,
        "gold_answer": example.final_answer,
        "gold_tool_ids": distinct_tool_ids,
        "gold_tool_sequence_ids": sequence_ids,
        "gold_dependency_edges": _dependency_edges(example, sequence_ids),
        "metadata": {
            "dataset": "traject_bench",
            "partition": example.partition,
            "trajectory_type": example.trajectory_type,
            "domain_name": example.domain_name,
            "raw_tool_call_count": len(sequence_ids),
            "distinct_gold_tool_count": len(distinct_tool_ids),
            "repeated_tool_call_count": len(sequence_ids) - len(distinct_tool_ids),
        },
    }
    return ConvertedTrajectBenchExample(
        ranking_record=ranking_record,
        label_record=label_record,
    )


def convert_traject_bench_examples(
    examples: Sequence[TrajectBenchExample],
    catalogs_by_domain: dict[str, PreparedTrajectBenchToolCatalog],
) -> TrajectBenchConversionResult:
    converted: list[ConvertedTrajectBenchExample] = []
    for example in examples:
        catalog = catalogs_by_domain.get(example.domain_name)
        if catalog is None:
            raise ValueError(
                f"Missing TRAJECT-Bench catalog for domain={example.domain_name!r}."
            )
        converted.append(convert_traject_bench_example(example, catalog))
    return TrajectBenchConversionResult(
        ranking_records=[item.ranking_record for item in converted],
        label_records=[item.label_record for item in converted],
    )


def combined_traject_bench_records(
    ranking_records: Sequence[TrajectBenchRankingRecord],
    label_records: Sequence[TrajectBenchLabelRecord],
) -> list[dict[str, object]]:
    labels_by_task_id = {label["task_id"]: label for label in label_records}
    combined: list[dict[str, object]] = []
    for ranking_record in ranking_records:
        task_id = ranking_record["task_id"]
        label = labels_by_task_id.get(task_id)
        if label is None:
            raise ValueError(f"Missing TRAJECT-Bench label for task_id={task_id}.")
        combined.append({**ranking_record, **label})
    return combined


def stable_traject_bench_tool_id(tool_name: str) -> NodeId:
    digest = hashlib.sha256(tool_name.encode("utf-8")).hexdigest()[:20]
    return f"t_{digest}"


def _merge_tool_definitions(
    definitions: Sequence[TrajectBenchToolDefinition],
) -> TrajectBenchToolDefinition:
    first = definitions[0]
    parameters_by_key: dict[tuple[str, bool], list[TrajectBenchToolParameter]] = defaultdict(list)
    for definition in definitions:
        for parameter in definition.parameters:
            parameters_by_key[(parameter.name, parameter.required)].append(parameter)
    parameters = tuple(
        _merge_parameters(parameters_by_key[key])
        for key in sorted(parameters_by_key, key=lambda value: (not value[1], *_text_sort_key(value[0])))
    )
    connected_names = tuple(
        sorted(
            {
                name
                for definition in definitions
                for name in definition.connected_tool_names
                if name != first.tool_name
            },
            key=_text_sort_key,
        )
    )
    return TrajectBenchToolDefinition(
        tool_name=first.tool_name,
        tool_description=_most_informative(
            definition.tool_description for definition in definitions
        ),
        parent_tool_name=_most_informative(
            definition.parent_tool_name for definition in definitions
        ),
        parent_tool_description=_most_informative(
            definition.parent_tool_description for definition in definitions
        ),
        api_name=_most_informative(definition.api_name for definition in definitions),
        domain_name=first.domain_name,
        category=_most_informative(definition.category for definition in definitions),
        parameters=parameters,
        connected_tool_names=connected_names,
    )


def _merge_parameters(
    parameters: Sequence[TrajectBenchToolParameter],
) -> TrajectBenchToolParameter:
    first = parameters[0]
    return TrajectBenchToolParameter(
        name=first.name,
        parameter_type=_most_informative(
            parameter.parameter_type for parameter in parameters
        ),
        description=_most_informative(
            parameter.description for parameter in parameters
        ),
        required=first.required,
    )


def _candidate_text(tool: TrajectBenchToolDefinition) -> str:
    sections = [
        f"Tool: {tool.tool_name}",
        f"Provider: {tool.parent_tool_name}",
        f"API: {tool.api_name}",
    ]
    if tool.tool_description:
        sections.append(f"Description: {tool.tool_description}")
    if tool.parent_tool_description:
        sections.append(f"Provider description: {tool.parent_tool_description}")
    required = [parameter for parameter in tool.parameters if parameter.required]
    optional = [parameter for parameter in tool.parameters if not parameter.required]
    if required:
        sections.append(f"Required parameters: {_parameter_text(required)}")
    if optional:
        sections.append(f"Optional parameters: {_parameter_text(optional)}")
    return "\n".join(sections)


def _parameter_text(parameters: Sequence[TrajectBenchToolParameter]) -> str:
    values: list[str] = []
    for parameter in parameters:
        value = parameter.name
        if parameter.parameter_type:
            value += f" ({parameter.parameter_type})"
        if parameter.description:
            value += f": {parameter.description}"
        values.append(value)
    return "; ".join(values)


def _dependency_edges(
    example: TrajectBenchExample,
    sequence_ids: Sequence[NodeId],
) -> list[list[NodeId]]:
    if example.trajectory_type == "parallel":
        return []
    edges: list[list[NodeId]] = []
    seen: set[tuple[NodeId, NodeId]] = set()
    for source, target in zip(sequence_ids, sequence_ids[1:]):
        edge = (source, target)
        if source == target or edge in seen:
            continue
        seen.add(edge)
        edges.append([source, target])
    return edges


def _task_id(example: TrajectBenchExample) -> TaskId:
    domain = _slug(example.domain_name)
    source_stem = _slug(example.source_key.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    return (
        f"traject_bench_{example.partition}_{domain}_{source_stem}_"
        f"{example.source_index:04d}"
    )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Cannot derive TRAJECT-Bench identifier from {value!r}.")
    return slug


def _most_informative(values: Iterable[str]) -> str:
    materialized = list(values)
    return max(materialized, key=lambda value: (len(value), value))


def _text_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


__all__ = [
    "MissingCatalogToolsError",
    "canonicalize_traject_bench_tool_catalog",
    "combined_traject_bench_records",
    "convert_traject_bench_example",
    "convert_traject_bench_examples",
    "prepare_traject_bench_tool_catalog",
    "stable_traject_bench_tool_id",
]
