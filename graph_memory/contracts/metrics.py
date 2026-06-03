from __future__ import annotations

from typing import TypeAlias, TypedDict

from graph_memory.contracts.common import MethodName, NodeId, TaskId

MetricValue: TypeAlias = str | float

MetricRow = TypedDict(
    "MetricRow",
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

MetricTableRow: TypeAlias = dict[str, MetricValue]

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


class FailureCase(TypedDict):
    debug_type: str
    task_id: TaskId
    method: MethodName
    failure_type: str
    gold_evidence_nodes: list[NodeId]
    retrieved_top_k: list[NodeId]
    missing_gold_nodes: list[NodeId]
    connected_gold_in_top_k: bool


__all__ = ["FailureCase", "MetricRow", "MetricTableRow", "MetricValue", "TaskMetricRow"]
