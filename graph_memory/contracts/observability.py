from __future__ import annotations

from typing import NotRequired, TypedDict

from graph_memory.contracts.common import JsonObject, MethodName, TaskId
from graph_memory.contracts.ranking import RankedNodeRecord


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


class RankedNodeDebugRecord(RankedNodeRecord, total=False):
    score_components: object


class ScoreDebugRecord(TypedDict, total=False):
    debug_type: str
    task_id: TaskId
    method: MethodName
    top_k: int
    ranked_nodes: list[RankedNodeDebugRecord]
    split: str
    config_digest: str


__all__ = ["GraphStatistics", "RankedNodeDebugRecord", "RunSummary", "ScoreDebugRecord"]
