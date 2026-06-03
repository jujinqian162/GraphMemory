from __future__ import annotations

from graph_memory.contracts.common import ALLOWED_EDGE_TYPES, ALLOWED_NODE_TYPES
from graph_memory.validation.common import (
    ContractValidationError,
    ValidationRecord,
    _memory_node_ids,
    _reject_unknown_fields,
    _require_record_list,
    _require_record_map,
    _require_unique,
    _required_finite_number,
    _required_string,
    validate_no_label_fields,
)
from graph_memory.validation.tasks import MEMORY_ITEM_FIELDS

GRAPH_FIELDS = {"task_id", "nodes", "edges", "metadata", "debug"}
QUESTION_NODE_FIELDS = {"id", "node_type", "text"}
GRAPH_EDGE_FIELDS = {"source", "target", "edge_type", "weight", "directed"}


def validate_graphs(graphs: object, inputs_by_task_id: object) -> None:
    graphs = _require_record_list(graphs, "graphs")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")

    seen_task_ids: set[str] = set()
    for index, graph in enumerate(graphs):
        if not isinstance(graph, dict):
            raise ContractValidationError(f"Invalid graph: record index={index} is not an object.")
        task_id = _required_string(graph, "task_id", "graph")
        _reject_unknown_fields(graph, GRAPH_FIELDS, "graph", task_id)
        validate_no_label_fields(graph, artifact_name="graph", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "graph task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid graph: task_id={task_id} has no matching input task.")

        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} nodes must be a list.")
        if not isinstance(edges, list):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} edges must be a list.")

        graph_node_ids = _validate_graph_nodes(nodes, inputs_by_task_id[task_id], task_id)
        for edge in edges:
            _validate_graph_edge(edge, graph_node_ids, task_id)


def _validate_graph_nodes(nodes: list[object], task_input: ValidationRecord, task_id: str) -> set[str]:
    seen_node_ids: set[str] = set()
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
            _reject_unknown_fields(node, MEMORY_ITEM_FIELDS, "memory graph node", task_id)
            if node_type != "document_sentence":
                raise ContractValidationError(
                    f"Invalid graph: task_id={task_id} node_id={node_id} memory node must have node_type=document_sentence."
                )

    if question_count != 1:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} must contain exactly one q node.")

    expected_memory_node_ids = _memory_node_ids(task_input)
    observed_memory_node_ids = seen_node_ids - {"q"}
    if observed_memory_node_ids != expected_memory_node_ids:
        missing = sorted(expected_memory_node_ids - observed_memory_node_ids)
        extra = sorted(observed_memory_node_ids - expected_memory_node_ids)
        raise ContractValidationError(
            f"Invalid graph: task_id={task_id} graph memory nodes mismatch; missing={missing} extra={extra}."
        )
    return seen_node_ids


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


__all__ = ["validate_graphs"]
