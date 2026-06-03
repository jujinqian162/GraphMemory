"""Domain-owned artifact contracts."""

from graph_memory.contracts.common import (
    ALLOWED_EDGE_TYPES,
    ALLOWED_NODE_TYPES,
    NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES,
    NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES,
    TRAIN_PAIR_SAMPLE_TYPES,
    EdgeType,
    JsonArray,
    JsonObject,
    JsonValue,
    MethodName,
    NodeId,
    NodeType,
    Score,
    TaskId,
    TrainPairSampleType,
)
from graph_memory.contracts.graphs import GraphEdge, GraphMemoryNode, GraphNode, MemoryGraph, QuestionNode
from graph_memory.contracts.metrics import FailureCase, MetricRow, MetricTableRow, MetricValue, TaskMetricRow
from graph_memory.contracts.observability import GraphStatistics, RankedNodeDebugRecord, RunSummary, ScoreDebugRecord
from graph_memory.contracts.ranking import RankedNodeRecord, RankedResult, RetrievedSubgraph
from graph_memory.contracts.tasks import CombinedMemoryTask, MemoryItem, MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairBuildSummary, TrainPairRecord

__all__ = [
    "ALLOWED_EDGE_TYPES",
    "ALLOWED_NODE_TYPES",
    "CombinedMemoryTask",
    "EdgeType",
    "FailureCase",
    "GraphEdge",
    "GraphMemoryNode",
    "GraphNode",
    "GraphStatistics",
    "JsonArray",
    "JsonObject",
    "JsonValue",
    "MemoryGraph",
    "MemoryItem",
    "MemoryTaskInput",
    "MemoryTaskLabels",
    "MethodName",
    "MetricRow",
    "MetricTableRow",
    "MetricValue",
    "NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES",
    "NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES",
    "NodeId",
    "NodeType",
    "QuestionNode",
    "RankedNodeDebugRecord",
    "RankedNodeRecord",
    "RankedResult",
    "RetrievedSubgraph",
    "RunSummary",
    "Score",
    "ScoreDebugRecord",
    "TRAIN_PAIR_SAMPLE_TYPES",
    "TaskId",
    "TaskMetricRow",
    "TrainPairBuildSummary",
    "TrainPairRecord",
    "TrainPairSampleType",
]
