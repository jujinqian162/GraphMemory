from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictStr, StringConstraints, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr, reject_label_fields
from graph_memory.trajectories import SourceSpan

NamespacedIdentifier = Annotated[
    StrictStr,
    StringConstraints(
        min_length=3,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$",
    ),
]

TOOL_CALL_NODE = "execution.tool_call"
TOOL_OUTPUT_NODE = "execution.tool_output"
ARTIFACT_NODE = "resource.artifact"

RETURNS_EDGE = "execution.returns"
PRECEDES_EDGE = "temporal.precedes"
FEEDS_EDGE = "data.feeds"
READS_EDGE = "resource.reads"
WRITES_EDGE = "resource.writes"

CORE_NODE_KINDS = frozenset({TOOL_CALL_NODE, TOOL_OUTPUT_NODE, ARTIFACT_NODE})
CORE_RELATIONS = frozenset(
    {RETURNS_EDGE, PRECEDES_EDGE, FEEDS_EDGE, READS_EDGE, WRITES_EDGE}
)
_CORE_NODE_NAMESPACES = frozenset({"execution", "resource"})
_CORE_RELATION_NAMESPACES = frozenset(
    {"execution", "temporal", "data", "resource"}
)
_CORE_ENDPOINTS = {
    RETURNS_EDGE: (TOOL_CALL_NODE, TOOL_OUTPUT_NODE),
    PRECEDES_EDGE: (TOOL_CALL_NODE, TOOL_CALL_NODE),
    FEEDS_EDGE: (TOOL_OUTPUT_NODE, TOOL_CALL_NODE),
    READS_EDGE: (TOOL_CALL_NODE, ARTIFACT_NODE),
    WRITES_EDGE: (TOOL_CALL_NODE, ARTIFACT_NODE),
}


class ProvenanceNode(DomainModel):
    node_id: NonEmptyStr
    kind: NamespacedIdentifier
    text: str
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    attributes: dict[str, JsonValue] | None = None


class ProvenanceEdge(DomainModel):
    edge_id: NonEmptyStr
    relation: NamespacedIdentifier
    source: NonEmptyStr
    target: NonEmptyStr
    derivation: Literal["native", "deterministic", "annotator"]
    extractor: NonEmptyStr
    source_spans: tuple[SourceSpan, ...] = ()
    attributes: dict[str, JsonValue] | None = None


class ProvenanceGraph(DomainModel):
    schema_version: Literal[1] = 1
    graph_id: NonEmptyStr
    trajectory_fingerprint: Annotated[
        StrictStr, StringConstraints(pattern=r"^[a-f0-9]{64}$")
    ]
    nodes: tuple[ProvenanceNode, ...]
    edges: tuple[ProvenanceEdge, ...]
    metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _validate_graph(self) -> "ProvenanceGraph":
        node_by_id = {node.node_id: node for node in self.nodes}
        if len(node_by_id) != len(self.nodes):
            raise ValueError("provenance graph node IDs must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("provenance graph edge IDs must be unique")

        for node in self.nodes:
            namespace = node.kind.split(".", 1)[0]
            if namespace in _CORE_NODE_NAMESPACES and node.kind not in CORE_NODE_KINDS:
                raise ValueError(f"unsupported core node kind: {node.kind}")
        for edge in self.edges:
            if edge.source not in node_by_id or edge.target not in node_by_id:
                raise ValueError(
                    f"edge={edge.edge_id} references a missing endpoint"
                )
            namespace = edge.relation.split(".", 1)[0]
            if (
                namespace in _CORE_RELATION_NAMESPACES
                and edge.relation not in CORE_RELATIONS
            ):
                raise ValueError(f"unsupported core relation: {edge.relation}")
            expected = _CORE_ENDPOINTS.get(edge.relation)
            if expected is not None:
                observed = (
                    node_by_id[edge.source].kind,
                    node_by_id[edge.target].kind,
                )
                if observed != expected:
                    detail = (
                        f"relation={edge.relation} expects endpoints={expected}, got={observed}"
                    )
                    raise ValueError(detail)

        reject_label_fields(
            self.model_dump(mode="python", exclude_none=True),
            path="provenance graph",
        )
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def node_by_id(self) -> dict[str, ProvenanceNode]:
        return {node.node_id: node for node in self.nodes}


__all__ = [
    "ARTIFACT_NODE",
    "CORE_NODE_KINDS",
    "CORE_RELATIONS",
    "FEEDS_EDGE",
    "NamespacedIdentifier",
    "PRECEDES_EDGE",
    "ProvenanceEdge",
    "ProvenanceGraph",
    "ProvenanceNode",
    "READS_EDGE",
    "RETURNS_EDGE",
    "TOOL_CALL_NODE",
    "TOOL_OUTPUT_NODE",
    "WRITES_EDGE",
]
