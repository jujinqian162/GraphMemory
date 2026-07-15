from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from graph_memory.compat import StrEnum
from graph_memory.contracts.common import JsonValue, TaskId


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


@dataclass(frozen=True)
class FieldBinding:
    output_field: str
    input_parameter: str
    binding_value_hash: str
    binding_kind: str

    def __post_init__(self) -> None:
        for field_name in (
            "output_field",
            "input_parameter",
            "binding_value_hash",
            "binding_kind",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"Field binding {field_name} must be non-empty.")


@dataclass(frozen=True)
class ExecutionProvenanceNode:
    node_id: str
    node_type: ProvenanceNodeType
    text: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("Execution provenance node_id must be non-empty.")


@dataclass(frozen=True)
class ExecutionProvenanceEdge:
    source: str
    target: str
    edge_type: ProvenanceEdgeType
    binding: FieldBinding | None = None
    weight: float = 1.0
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("Execution provenance edge endpoints must be non-empty.")
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(
                "Execution provenance edge weight must be finite and non-negative."
            )
        if self.edge_type is ProvenanceEdgeType.FEEDS and self.binding is None:
            raise ValueError("feeds edge requires explicit field binding metadata.")
        if self.edge_type is not ProvenanceEdgeType.FEEDS and self.binding is not None:
            raise ValueError("Field binding metadata is only valid on feeds edges.")


@dataclass(frozen=True)
class ExecutionProvenanceGraph:
    task_id: TaskId
    nodes: tuple[ExecutionProvenanceNode, ...]
    edges: tuple[ExecutionProvenanceEdge, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("Execution provenance graph task_id must be non-empty.")
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("Execution provenance graph node IDs must be unique.")
        for edge in self.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                raise ValueError(
                    f"Execution provenance edge references missing node: {edge.source}->{edge.target}."
                )
            _validate_edge_transition(edge, node_by_id)

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
            f"Invalid {edge.edge_type.value} transition: self loops are not allowed."
        )
    allowed = expected[edge.edge_type]
    allowed_sources, allowed_targets = allowed
    if source_type not in allowed_sources or target_type not in allowed_targets:
        raise ValueError(
            f"Invalid {edge.edge_type.value} transition: "
            f"{source_type.value}->{target_type.value}."
        )


__all__ = [
    "ExecutionProvenanceEdge",
    "ExecutionProvenanceGraph",
    "ExecutionProvenanceNode",
    "FieldBinding",
    "ProvenanceEdgeType",
    "ProvenanceNodeType",
]
