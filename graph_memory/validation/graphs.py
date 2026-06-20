from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from graph_memory.contracts.common import ALLOWED_EDGE_TYPES, ALLOWED_NODE_TYPES
from graph_memory.graphs.requests import GraphBuildRequest
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.validation.common import (
    ContractValidationError,
    ValidationRecord,
    _reject_unknown_fields,
    _require_record_list,
    _require_unique,
    _required_finite_number,
    _required_int,
    _required_string,
    validate_no_label_fields,
)

GRAPH_ITEM_NODE_FIELDS = {"id", "node_type", "node_kind", "text", "source_ref", "group_key", "sequence_index", "metadata"}
GRAPH_FIELDS = {"task_id", "nodes", "edges", "metadata", "debug"}
QUESTION_NODE_FIELDS = {"id", "node_type", "text"}
GRAPH_EDGE_FIELDS = {"source", "target", "edge_type", "weight", "directed"}


def validate_graphs(graphs: object, expected_items_by_task_id: object) -> None:
    graphs = _require_record_list(graphs, "graphs")
    expected_by_task_id = _expected_item_ids_by_task_id(expected_items_by_task_id)

    seen_task_ids: set[str] = set()
    for index, graph in enumerate(graphs):
        if not isinstance(graph, dict):
            raise ContractValidationError(f"Invalid graph: record index={index} is not an object.")
        task_id = _required_string(graph, "task_id", "graph")
        _reject_unknown_fields(graph, GRAPH_FIELDS, "graph", task_id)
        validate_no_label_fields(graph, artifact_name="graph", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "graph task_id")
        if task_id not in expected_by_task_id:
            raise ContractValidationError(f"Invalid graph: task_id={task_id} has no expected graph item ids.")

        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} nodes must be a list.")
        if not isinstance(edges, list):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} edges must be a list.")

        graph_node_ids, graph_item_ids = _validate_graph_nodes(nodes, task_id)
        expected_item_ids = expected_by_task_id[task_id]
        if graph_item_ids != expected_item_ids:
            missing = sorted(expected_item_ids - graph_item_ids)
            extra = sorted(graph_item_ids - expected_item_ids)
            raise ContractValidationError(
                f"Invalid graph: task_id={task_id} graph item ids mismatch; missing={missing} extra={extra}."
            )
        for edge in edges:
            _validate_graph_edge(edge, graph_node_ids, task_id)


def expected_item_ids_from_text_requests(requests: Sequence[TextRankingRequest]) -> dict[str, set[str]]:
    return {request.task_id: {candidate.item_id for candidate in request.candidates} for request in requests}


def expected_item_ids_from_graph_requests(requests: Sequence[GraphBuildRequest]) -> dict[str, set[str]]:
    return {request.task_id: {node.node_id for node in request.nodes} for request in requests}


def _expected_item_ids_by_task_id(value: object) -> dict[str, set[str]]:
    if isinstance(value, Mapping):
        expected: dict[str, set[str]] = {}
        for task_id, raw_ids in value.items():
            if not isinstance(task_id, str) or not task_id:
                raise ContractValidationError("Invalid expected graph item ids: task ids must be non-empty strings.")
            expected[task_id] = _string_id_set(raw_ids, artifact_name="expected graph item ids", task_id=task_id)
        return expected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(isinstance(item, GraphBuildRequest) for item in value):
            return expected_item_ids_from_graph_requests(cast(Sequence[GraphBuildRequest], value))
        if all(isinstance(item, TextRankingRequest) for item in value):
            return expected_item_ids_from_text_requests(cast(Sequence[TextRankingRequest], value))
    raise ContractValidationError("Invalid expected graph item ids: expected mapping or request sequence.")


def _string_id_set(value: object, *, artifact_name: str, task_id: str) -> set[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} ids must be a sequence.")
    ids: set[str] = set()
    for item_id in value:
        if not isinstance(item_id, str) or not item_id:
            raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} ids must be non-empty strings.")
        ids.add(item_id)
    return ids


def _validate_graph_nodes(nodes: list[object], task_id: str) -> tuple[set[str], set[str]]:
    seen_node_ids: set[str] = set()
    graph_item_ids: set[str] = set()
    question_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} node is not an object.")
        node_id = _required_string(node, "id", "graph node", task_id)
        _require_unique(node_id, seen_node_ids, f"graph node id task_id={task_id}")
        node_type = _required_string(node, "node_type", "graph node", task_id)
        if node_type not in ALLOWED_NODE_TYPES:
            raise ContractValidationError(f"Invalid graph: task_id={task_id} node_id={node_id} unsupported node_type={node_type}.")
        if node_id == "q":
            question_count += 1
            _reject_unknown_fields(node, QUESTION_NODE_FIELDS, "question graph node", task_id)
            if node_type != "question":
                raise ContractValidationError(f"Invalid graph: task_id={task_id} q node must have node_type=question.")
            _required_string(node, "text", "question graph node", task_id)
        else:
            _reject_unknown_fields(node, GRAPH_ITEM_NODE_FIELDS, "graph item node", task_id)
            if node_type != "graph_item":
                raise ContractValidationError(
                    f"Invalid graph: task_id={task_id} node_id={node_id} non-query node must have node_type=graph_item."
                )
            _required_string(node, "node_kind", "graph item node", task_id)
            _required_string(node, "text", "graph item node", task_id)
            _optional_string(node, "source_ref", "graph item node", task_id)
            _optional_string(node, "group_key", "graph item node", task_id)
            if "sequence_index" in node:
                _required_int(node, "sequence_index", "graph item node", task_id, minimum=0)
            if "metadata" in node and not isinstance(node["metadata"], dict):
                raise ContractValidationError(f"Invalid graph: task_id={task_id} node_id={node_id} metadata must be an object.")
            graph_item_ids.add(node_id)

    if question_count != 1:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} must contain exactly one q node.")
    return seen_node_ids, graph_item_ids


def _optional_string(record: ValidationRecord, field_name: str, artifact_name: str, task_id: str) -> None:
    if field_name in record and not isinstance(record[field_name], str):
        raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} field={field_name} must be a string.")


def _validate_graph_edge(edge: ValidationRecord, graph_node_ids: set[str], task_id: str) -> None:
    if not isinstance(edge, dict):
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge is not an object.")
    _reject_unknown_fields(edge, GRAPH_EDGE_FIELDS, "graph edge", task_id)
    source = _required_string(edge, "source", "graph edge", task_id)
    target = _required_string(edge, "target", "graph edge", task_id)
    if source not in graph_node_ids:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge source={source} does not exist in nodes.")
    if target not in graph_node_ids:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge target={target} does not exist in nodes.")
    edge_type = _required_string(edge, "edge_type", "graph edge", task_id)
    if edge_type not in ALLOWED_EDGE_TYPES:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} unsupported edge_type={edge_type}.")
    _required_finite_number(edge, "weight", "graph edge", task_id, minimum=0.0)
    if not isinstance(edge.get("directed"), bool):
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge directed must be boolean.")


__all__ = ["expected_item_ids_from_graph_requests", "expected_item_ids_from_text_requests", "validate_graphs"]
