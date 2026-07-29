from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field, JsonValue, model_validator

from graph_memory.compat import StrEnum
from graph_memory.contracts.model import (
    DomainModel,
    NonEmptyStr,
    NonNegativeFiniteFloat,
)


class ProvenanceNodeType(StrEnum):
    TASK = "task"
    AGENT = "agent"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    ANSWER = "answer"
    OBSERVATION = "observation"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    VERIFICATION = "verification"
    DECISION = "decision"


class ProvenanceEdgeType(StrEnum):
    CONTAINS = "contains"
    INVOKES = "invokes"
    RETURNS = "returns"
    FEEDS = "feeds"
    GROUNDS = "grounds"
    PRECEDES = "precedes"
    SUPPORTS = "supports"
    VERIFIES = "verifies"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    INVALIDATES = "invalidates"
    AFFECTS = "affects"
    SUPERSEDES = "supersedes"


class FieldBinding(DomainModel):
    output_field: NonEmptyStr
    input_parameter: NonEmptyStr
    binding_value_hash: NonEmptyStr
    binding_kind: NonEmptyStr


class ExecutionProvenanceNode(DomainModel):
    node_id: NonEmptyStr
    node_type: ProvenanceNodeType
    text: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ExecutionProvenanceEdge(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr
    edge_type: ProvenanceEdgeType
    binding: FieldBinding | None = None
    weight: NonNegativeFiniteFloat = 1.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_binding(self) -> "ExecutionProvenanceEdge":
        if self.edge_type is ProvenanceEdgeType.FEEDS and self.binding is None:
            raise ValueError("feeds edge requires explicit field binding metadata")
        if self.edge_type is not ProvenanceEdgeType.FEEDS and self.binding is not None:
            raise ValueError("field binding metadata is only valid on feeds edges")
        return self


class ExecutionProvenanceGraph(DomainModel):
    task_id: NonEmptyStr
    nodes: tuple[ExecutionProvenanceNode, ...]
    edges: tuple[ExecutionProvenanceEdge, ...]

    @model_validator(mode="after")
    def _validate_graph(self) -> "ExecutionProvenanceGraph":
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("execution provenance graph node IDs must be unique")
        seen_edges: set[tuple[str, str, ProvenanceEdgeType]] = set()
        for edge in self.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                raise ValueError(
                    "execution provenance edge references missing node: "
                    f"{edge.source}->{edge.target}"
                )
            edge_key = (edge.source, edge.target, edge.edge_type)
            if edge_key in seen_edges:
                raise ValueError(
                    "execution provenance graph contains a duplicate typed edge: "
                    f"{edge.source}->{edge.target}:{edge.edge_type.value}"
                )
            seen_edges.add(edge_key)
            _validate_edge_transition(edge, node_by_id)
        return self

    def node(self, node_id: str) -> ExecutionProvenanceNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)


def _validate_edge_transition(
    edge: ExecutionProvenanceEdge,
    node_by_id: Mapping[str, ExecutionProvenanceNode],
) -> None:
    source_type = node_by_id[edge.source].node_type
    target_type = node_by_id[edge.target].node_type
    execution_nodes = {
        ProvenanceNodeType.TOOL_CALL,
        ProvenanceNodeType.TOOL_OUTPUT,
        ProvenanceNodeType.OBSERVATION,
        ProvenanceNodeType.EVIDENCE,
        ProvenanceNodeType.CLAIM,
        ProvenanceNodeType.VERIFICATION,
        ProvenanceNodeType.DECISION,
        ProvenanceNodeType.ANSWER,
    }
    support_sources = {
        ProvenanceNodeType.TOOL_OUTPUT,
        ProvenanceNodeType.OBSERVATION,
        ProvenanceNodeType.EVIDENCE,
        ProvenanceNodeType.CLAIM,
        ProvenanceNodeType.VERIFICATION,
    }
    support_targets = {
        ProvenanceNodeType.CLAIM,
        ProvenanceNodeType.ANSWER,
        ProvenanceNodeType.DECISION,
    }
    revision_nodes = support_sources | {
        ProvenanceNodeType.DECISION,
        ProvenanceNodeType.ANSWER,
    }
    expected: dict[
        ProvenanceEdgeType, tuple[set[ProvenanceNodeType], set[ProvenanceNodeType]]
    ] = {
        ProvenanceEdgeType.CONTAINS: (
            {ProvenanceNodeType.TASK},
            set(ProvenanceNodeType) - {ProvenanceNodeType.TASK},
        ),
        ProvenanceEdgeType.INVOKES: (
            {ProvenanceNodeType.AGENT},
            {ProvenanceNodeType.TOOL_CALL},
        ),
        ProvenanceEdgeType.RETURNS: (
            {ProvenanceNodeType.TOOL_CALL},
            {ProvenanceNodeType.TOOL_OUTPUT, ProvenanceNodeType.OBSERVATION},
        ),
        ProvenanceEdgeType.FEEDS: (
            {
                ProvenanceNodeType.TOOL_OUTPUT,
                ProvenanceNodeType.OBSERVATION,
                ProvenanceNodeType.EVIDENCE,
            },
            {ProvenanceNodeType.TOOL_CALL},
        ),
        ProvenanceEdgeType.GROUNDS: (
            {
                ProvenanceNodeType.TOOL_OUTPUT,
                ProvenanceNodeType.OBSERVATION,
                ProvenanceNodeType.EVIDENCE,
            },
            support_targets,
        ),
        ProvenanceEdgeType.PRECEDES: (execution_nodes, execution_nodes),
        ProvenanceEdgeType.SUPPORTS: (support_sources, support_targets),
        ProvenanceEdgeType.VERIFIES: (
            {ProvenanceNodeType.VERIFICATION},
            support_targets,
        ),
        ProvenanceEdgeType.DEPENDS_ON: (execution_nodes, execution_nodes),
        ProvenanceEdgeType.CONTRADICTS: (support_sources, support_targets),
        ProvenanceEdgeType.INVALIDATES: (revision_nodes, revision_nodes),
        ProvenanceEdgeType.SUPERSEDES: (revision_nodes, revision_nodes),
        ProvenanceEdgeType.AFFECTS: (
            revision_nodes,
            {
                ProvenanceNodeType.TOOL_CALL,
                ProvenanceNodeType.CLAIM,
                ProvenanceNodeType.ANSWER,
                ProvenanceNodeType.DECISION,
            },
        ),
    }
    if edge.source == edge.target:
        raise ValueError(
            f"invalid {edge.edge_type.value} transition: self loops are not allowed"
        )
    allowed_sources, allowed_targets = expected[edge.edge_type]
    if source_type not in allowed_sources or target_type not in allowed_targets:
        raise ValueError(
            f"invalid {edge.edge_type.value} transition: "
            f"{source_type.value}->{target_type.value}"
        )


__all__ = [
    "ExecutionProvenanceEdge",
    "ExecutionProvenanceGraph",
    "ExecutionProvenanceNode",
    "FieldBinding",
    "ProvenanceEdgeType",
    "ProvenanceNodeType",
]
