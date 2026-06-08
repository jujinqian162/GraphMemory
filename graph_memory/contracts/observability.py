from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired

from graph_memory.contracts.common import JsonObject


class GraphStatistics(TypedDict):
    num_graphs: int
    avg_nodes: float
    avg_edges: float
    edge_counts_by_type: dict[str, int]
    isolated_memory_nodes: int
    split: NotRequired[str]
    graph_config: NotRequired[JsonObject]


class RunSummary(TypedDict):
    script: str
    started_at: str
    finished_at: str
    status: str
    effective_config: JsonObject
    inputs: JsonObject
    outputs: JsonObject
    counts: JsonObject
    timings: JsonObject
    environment: dict[str, str]
    notes: list[str]
    error: NotRequired[str]


__all__ = ["GraphStatistics", "RunSummary"]
