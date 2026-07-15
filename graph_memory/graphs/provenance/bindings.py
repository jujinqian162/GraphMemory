from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeGuard

from graph_memory.graphs.provenance.contracts import (
    ExecutionProvenanceEdge,
    ExecutionProvenanceNode,
    ProvenanceEdgeType,
)


def binding_matches_endpoints(
    edge: ExecutionProvenanceEdge,
    node_by_id: Mapping[str, ExecutionProvenanceNode],
) -> bool:
    if edge.edge_type is not ProvenanceEdgeType.FEEDS or edge.binding is None:
        return False
    source = node_by_id.get(edge.source)
    target = node_by_id.get(edge.target)
    if source is None or target is None:
        return False
    output_fields = source.metadata.get("output_field_hashes")
    input_parameters = target.metadata.get("input_parameters")
    if not isinstance(output_fields, Mapping) or not _string_sequence(
        input_parameters
    ):
        return False
    observed_hash = output_fields.get(edge.binding.output_field)
    return (
        isinstance(observed_hash, str)
        and observed_hash == edge.binding.binding_value_hash
        and edge.binding.input_parameter in input_parameters
    )


def binding_relation_key(edge: ExecutionProvenanceEdge) -> str:
    if edge.edge_type is not ProvenanceEdgeType.FEEDS or edge.binding is None:
        raise ValueError("Only bound feeds edges have a binding relation key.")
    binding = edge.binding
    return ":".join(
        (
            ProvenanceEdgeType.FEEDS.value,
            binding.output_field,
            binding.input_parameter,
            binding.binding_kind,
        )
    )


def _string_sequence(value: object) -> TypeGuard[Sequence[str]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, str) and item for item in value)
    )


__all__ = ["binding_matches_endpoints", "binding_relation_key"]
