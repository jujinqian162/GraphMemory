from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, StrictBool, field_validator, model_validator

from graph_memory.contracts.common import EdgeType
from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    reject_label_fields,
)


class QuestionNode(DomainModel):
    id: Literal["q"] = "q"
    node_type: Literal["question"] = "question"
    text: str


class GraphItemNode(DomainModel):
    id: NonEmptyStr
    node_type: Literal["graph_item"] = "graph_item"
    node_kind: NonEmptyStr
    text: str
    source_ref: str | None = None
    group_key: str | None = None
    sequence_index: NonNegativeInt | None = None
    metadata: dict[str, JsonValue] | None = None

    @field_validator("id")
    @classmethod
    def _reserve_query_id(cls, value: str) -> str:
        if value == "q":
            raise ValueError("graph item id 'q' is reserved for the question node")
        return value


GraphNode: TypeAlias = Annotated[
    QuestionNode | GraphItemNode, Field(discriminator="node_type")
]


class GraphEdge(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: EdgeType
    weight: NonNegativeFiniteFloat
    directed: StrictBool


class EvidenceGraph(DomainModel):
    task_id: NonEmptyStr
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    metadata: dict[str, JsonValue] | None = None
    debug: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_graph(self) -> "EvidenceGraph":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node IDs must be unique")
        if sum(isinstance(node, QuestionNode) for node in self.nodes) != 1:
            raise ValueError("graph must contain exactly one question node")
        node_id_set = set(node_ids)
        for edge in self.edges:
            if edge.source not in node_id_set:
                raise ValueError(f"edge source={edge.source} does not exist in nodes")
            if edge.target not in node_id_set:
                raise ValueError(f"edge target={edge.target} does not exist in nodes")
        reject_label_fields(self.model_dump(mode="python", exclude_none=True), path="graph")
        return self

    @property
    def graph_item_ids(self) -> frozenset[str]:
        return frozenset(
            node.id for node in self.nodes if isinstance(node, GraphItemNode)
        )


class EvidenceGraphBatch(DomainModel):
    graphs: tuple[EvidenceGraph, ...]
    expected_item_ids_by_task_id: dict[NonEmptyStr, frozenset[NonEmptyStr]]

    @model_validator(mode="after")
    def _validate_coverage(self) -> "EvidenceGraphBatch":
        task_ids = [graph.task_id for graph in self.graphs]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("graph task IDs must be unique")
        observed = set(task_ids)
        expected = set(self.expected_item_ids_by_task_id)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"graph task coverage mismatch; missing={missing} extra={extra}"
            )
        for graph in self.graphs:
            expected_items = self.expected_item_ids_by_task_id[graph.task_id]
            if graph.graph_item_ids != expected_items:
                missing = sorted(expected_items - graph.graph_item_ids)
                extra = sorted(graph.graph_item_ids - expected_items)
                raise ValueError(
                    f"task_id={graph.task_id} graph item ids mismatch; "
                    f"missing={missing} extra={extra}"
                )
        return self


__all__ = [
    "EvidenceGraph",
    "EvidenceGraphBatch",
    "GraphEdge",
    "GraphItemNode",
    "GraphNode",
    "QuestionNode",
]
