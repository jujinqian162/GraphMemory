from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from graph_memory.contracts.common import JsonValue

TrajectBenchPartition = Literal[
    "parallel_simple",
    "parallel_hard",
    "sequential",
]
TrajectBenchTrajectoryType = Literal["parallel", "sequential"]


@dataclass(frozen=True)
class TrajectBenchToolParameter:
    name: str
    parameter_type: str
    description: str
    required: bool


@dataclass(frozen=True)
class TrajectBenchToolDefinition:
    tool_name: str
    tool_description: str
    parent_tool_name: str
    parent_tool_description: str
    api_name: str
    domain_name: str
    category: str
    parameters: tuple[TrajectBenchToolParameter, ...]
    connected_tool_names: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalTrajectBenchToolCatalog:
    domain_name: str
    tools: tuple[TrajectBenchToolDefinition, ...]
    duplicate_entry_count: int
    unresolved_connection_count: int


@dataclass(frozen=True)
class TrajectBenchExample:
    source_key: str
    source_index: int
    partition: TrajectBenchPartition
    domain_name: str
    query: str
    trajectory_type: TrajectBenchTrajectoryType
    tool_names: tuple[str, ...]
    final_answer: str


class TrajectBenchCandidateTool(TypedDict):
    tool_id: str
    tool_name: str
    text: str
    parent_tool_name: str
    api_name: str
    domain_name: str
    catalog_position: int
    connected_tool_ids: list[str]


class TrajectBenchRankingRecord(TypedDict):
    task_id: str
    query: str
    trajectory_type: TrajectBenchTrajectoryType
    domain_name: str
    candidate_tools: list[TrajectBenchCandidateTool]
    metadata: dict[str, JsonValue]


class TrajectBenchLabelRecord(TypedDict):
    task_id: str
    gold_answer: str
    gold_tool_ids: list[str]
    gold_tool_sequence_ids: list[str]
    gold_dependency_edges: list[list[str]]
    metadata: dict[str, JsonValue]


class CombinedTrajectBenchRecord(
    TrajectBenchRankingRecord,
    TrajectBenchLabelRecord,
):
    """Combined inspection artifact; retrieval code must not consume it."""


@dataclass(frozen=True)
class PreparedTrajectBenchToolCatalog:
    domain_name: str
    candidates: tuple[TrajectBenchCandidateTool, ...]
    tool_ids_by_name: dict[str, str]
    duplicate_entry_count: int
    unresolved_connection_count: int


@dataclass(frozen=True)
class ConvertedTrajectBenchExample:
    ranking_record: TrajectBenchRankingRecord
    label_record: TrajectBenchLabelRecord


@dataclass(frozen=True)
class TrajectBenchConversionResult:
    ranking_records: list[TrajectBenchRankingRecord]
    label_records: list[TrajectBenchLabelRecord]


__all__ = [
    "CanonicalTrajectBenchToolCatalog",
    "CombinedTrajectBenchRecord",
    "ConvertedTrajectBenchExample",
    "PreparedTrajectBenchToolCatalog",
    "TrajectBenchCandidateTool",
    "TrajectBenchConversionResult",
    "TrajectBenchExample",
    "TrajectBenchLabelRecord",
    "TrajectBenchPartition",
    "TrajectBenchRankingRecord",
    "TrajectBenchToolDefinition",
    "TrajectBenchToolParameter",
    "TrajectBenchTrajectoryType",
]
