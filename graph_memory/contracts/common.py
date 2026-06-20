from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

TaskId = str
NodeId = str
MethodName = str
Score = float
JsonObject: TypeAlias = Mapping[str, "JsonValue"]
JsonArray: TypeAlias = Sequence["JsonValue"]
JsonValue: TypeAlias = str | int | float | bool | None | JsonArray | JsonObject

NodeType = Literal["question", "graph_item"]
EdgeType = Literal["sequential", "query_overlap", "entity_overlap", "bridge"]
TrainPairSampleType = Literal["positive", "easy_random", "hard_bm25", "hard_dense", "hard_graph_neighbor"]

ALLOWED_NODE_TYPES: set[str] = {"question", "graph_item"}
ALLOWED_EDGE_TYPES: set[str] = {"sequential", "query_overlap", "entity_overlap", "bridge"}
NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES: set[str] = {"sequential", "entity_overlap", "bridge"}
TRAIN_PAIR_SAMPLE_TYPES: set[str] = {"positive", "easy_random", "hard_bm25", "hard_dense", "hard_graph_neighbor"}
NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES: set[str] = TRAIN_PAIR_SAMPLE_TYPES - {"positive"}

__all__ = [
    "ALLOWED_EDGE_TYPES",
    "ALLOWED_NODE_TYPES",
    "EdgeType",
    "JsonArray",
    "JsonObject",
    "JsonValue",
    "MethodName",
    "NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES",
    "NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES",
    "NodeId",
    "NodeType",
    "Score",
    "TRAIN_PAIR_SAMPLE_TYPES",
    "TaskId",
    "TrainPairSampleType",
]
