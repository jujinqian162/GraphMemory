from __future__ import annotations

from collections import defaultdict

from pydantic import Field, model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.provenance.contracts import (
    FEEDS_EDGE,
    READS_EDGE,
    RETURNS_EDGE,
    TOOL_CALL_NODE,
    WRITES_EDGE,
    NamespacedIdentifier,
    ProvenanceGraph,
    ProvenanceNode,
)

RESOURCE_FLOW_RELATION = "resource.flow"


class OutputDependency(DomainModel):
    source_output_id: NonEmptyStr
    target_output_id: NonEmptyStr
    relation: NamespacedIdentifier
    supporting_edge_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dependency(self) -> "OutputDependency":
        if self.source_output_id == self.target_output_id:
            raise ValueError("output dependency cannot be a self edge")
        if len(self.supporting_edge_ids) != len(set(self.supporting_edge_ids)):
            raise ValueError("supporting edge IDs must be unique")
        return self


def logical_output_dependencies(
    graph: ProvenanceGraph,
) -> tuple[OutputDependency, ...]:
    """Project query-independent physical provenance into output dependencies."""

    calls = {
        node.node_id: node for node in graph.nodes if node.kind == TOOL_CALL_NODE
    }
    output_for_call: dict[str, str] = {}
    return_edge_for_call: dict[str, str] = {}
    for edge in graph.edges:
        if edge.relation == RETURNS_EDGE:
            output_for_call[edge.source] = edge.target
            return_edge_for_call[edge.source] = edge.edge_id

    dependencies: dict[tuple[str, str, str], OutputDependency] = {}
    for edge in graph.edges:
        if edge.relation != FEEDS_EDGE:
            continue
        target_output_id = output_for_call.get(edge.target)
        target_return_edge = return_edge_for_call.get(edge.target)
        if target_output_id is None or target_return_edge is None:
            continue
        dependency = OutputDependency(
            source_output_id=edge.source,
            target_output_id=target_output_id,
            relation=FEEDS_EDGE,
            supporting_edge_ids=(edge.edge_id, target_return_edge),
        )
        dependencies[_dependency_key(dependency)] = dependency

    writes_by_artifact: dict[str, list[tuple[str, str]]] = defaultdict(list)
    reads_by_artifact: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph.edges:
        if edge.relation == WRITES_EDGE:
            writes_by_artifact[edge.target].append((edge.source, edge.edge_id))
        elif edge.relation == READS_EDGE:
            reads_by_artifact[edge.target].append((edge.source, edge.edge_id))

    for artifact_id in sorted(set(writes_by_artifact) & set(reads_by_artifact)):
        writers = sorted(
            writes_by_artifact[artifact_id], key=lambda item: _position(calls[item[0]])
        )
        readers = sorted(
            reads_by_artifact[artifact_id], key=lambda item: _position(calls[item[0]])
        )
        for reader_call_id, read_edge_id in readers:
            prior_writers = [
                item
                for item in writers
                if _position(calls[item[0]]) < _position(calls[reader_call_id])
            ]
            if not prior_writers:
                continue
            writer_call_id, write_edge_id = prior_writers[-1]
            source_output_id = output_for_call.get(writer_call_id)
            target_output_id = output_for_call.get(reader_call_id)
            writer_return_edge = return_edge_for_call.get(writer_call_id)
            reader_return_edge = return_edge_for_call.get(reader_call_id)
            if (
                source_output_id is None
                or target_output_id is None
                or writer_return_edge is None
                or reader_return_edge is None
            ):
                continue
            dependency = OutputDependency(
                source_output_id=source_output_id,
                target_output_id=target_output_id,
                relation=RESOURCE_FLOW_RELATION,
                supporting_edge_ids=(
                    writer_return_edge,
                    write_edge_id,
                    read_edge_id,
                    reader_return_edge,
                ),
            )
            dependencies[_dependency_key(dependency)] = dependency

    return tuple(
        dependencies[key]
        for key in sorted(dependencies)
    )


def _position(node: ProvenanceNode) -> tuple[int, int]:
    attributes = node.attributes or {}
    message_index = attributes.get("message_index")
    sub_index = attributes.get("sub_index")
    return (
        message_index if isinstance(message_index, int) else 0,
        sub_index if isinstance(sub_index, int) else 0,
    )


def _dependency_key(dependency: OutputDependency) -> tuple[str, str, str]:
    return (
        dependency.source_output_id,
        dependency.target_output_id,
        dependency.relation,
    )


__all__ = [
    "OutputDependency",
    "RESOURCE_FLOW_RELATION",
    "logical_output_dependencies",
]
