from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from graph_memory.contracts.common import MethodName, NodeId, TaskId

MetricValue: TypeAlias = str | float


@dataclass(frozen=True)
class MetricTableSchema:
    name: str
    main_columns: tuple[str, ...]
    path_columns: tuple[str, ...]
    efficiency_columns: tuple[str, ...]
    wide_columns: tuple[str, ...]


EvidenceMetricRow = TypedDict(
    "EvidenceMetricRow",
    {
        "Method": str,
        "Recall@2": float,
        "Recall@5": float,
        "Recall@10": float,
        "Evidence F1@5": float,
        "Evidence F1@10": float,
        "Full Support@5": float,
        "Full Support@10": float,
        "MRR": float,
        "Connected Evidence Recall@5": float,
        "Connected Evidence Recall@10": float,
        "Query-Evidence Connectivity@10": float,
        "Path Recall@10": MetricValue,
        "Edge Recall@10": MetricValue,
        "Retrieval Latency / Query": float,
        "Index Build Time": float,
        "Graph Construction Time": float,
        "Memory Size": float,
        "Avg Retrieved Nodes": float,
        "Avg Retrieved Edges": float,
    },
)

LongMemEvalMetricRow = TypedDict(
    "LongMemEvalMetricRow",
    {
        "Method": str,
        "Turn Recall@5": float,
        "Turn Recall@10": float,
        "Full Turn Support@10": float,
        "Session Recall@5": float,
        "Session Recall@10": float,
        "Full Session Support@10": float,
        "MRR": float,
        "Path Recall@10": MetricValue,
        "Edge Recall@10": MetricValue,
        "Retrieval Latency / Query": float,
        "Memory Size": float,
        "Avg Retrieved Nodes": float,
        "Avg Retrieved Edges": float,
    },
)

MetricRow: TypeAlias = EvidenceMetricRow
SuiteMetricRow: TypeAlias = EvidenceMetricRow | LongMemEvalMetricRow
MetricTableRow: TypeAlias = dict[str, MetricValue]
MetricSuiteRow: TypeAlias = dict[str, MetricValue]

TaskMetricRow = TypedDict(
    "TaskMetricRow",
    {
        "Recall@2": float,
        "Recall@5": float,
        "Recall@10": float,
        "Evidence F1@5": float,
        "Evidence F1@10": float,
        "Full Support@5": float,
        "Full Support@10": float,
        "MRR": float,
        "Connected Evidence Recall@5": float,
        "Connected Evidence Recall@10": float,
        "Query-Evidence Connectivity@10": float,
        "Retrieval Latency / Query": float,
        "Memory Size": float,
        "Avg Retrieved Nodes": float,
        "Avg Retrieved Edges": float,
    },
)


class FailureCase(TypedDict, total=False):
    debug_type: str
    task_id: TaskId
    method: MethodName
    failure_type: str
    gold_evidence_item_ids: list[NodeId]
    gold_support_item_ids: list[NodeId]
    gold_support_session_ids: list[str]
    retrieved_top_k: list[NodeId]
    retrieved_sessions_top_k: list[str]
    missing_gold_nodes: list[NodeId]
    connected_gold_in_top_k: bool


__all__ = [
    "EvidenceMetricRow",
    "FailureCase",
    "LongMemEvalMetricRow",
    "MetricRow",
    "MetricSuiteRow",
    "MetricTableRow",
    "MetricTableSchema",
    "MetricValue",
    "SuiteMetricRow",
    "TaskMetricRow",
]
