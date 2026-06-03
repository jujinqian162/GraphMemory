from __future__ import annotations

from typing import Any

from graph_memory.retrieval_registry import get_supported_methods
from graph_memory.validation.common import (
    ContractValidationError,
    _memory_node_ids,
    _reject_unknown_fields,
    _require_record_list,
    _require_record_map,
    _require_unique,
    _required_finite_number,
    _required_int,
    _required_string,
)
from graph_memory.validation.graphs import _validate_graph_edge

RANKED_RESULT_FIELDS = {
    "task_id",
    "method",
    "ranked_nodes",
    "retrieved_subgraph",
    "latency_ms",
    "input_tokens",
    "metadata",
    "debug",
}
RANKED_NODE_FIELDS = {"node_id", "score"}
RETRIEVED_SUBGRAPH_FIELDS = {"nodes", "edges"}


def validate_ranked_results(predictions: object, inputs_by_task_id: object) -> None:
    predictions = _require_record_list(predictions, "ranked results")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")

    seen_task_ids: set[str] = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ContractValidationError(f"Invalid ranked results: record index={index} is not an object.")
        task_id = _required_string(prediction, "task_id", "ranked result")
        _reject_unknown_fields(prediction, RANKED_RESULT_FIELDS, "ranked result", task_id)
        _require_unique(task_id, seen_task_ids, "ranked result task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} has no matching input task.")

        method = _required_string(prediction, "method", "ranked result", task_id)
        if method not in get_supported_methods():
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} unsupported method={method}.")

        ranked_nodes = prediction.get("ranked_nodes")
        if not isinstance(ranked_nodes, list):
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} ranked_nodes must be a list.")
        expected_node_ids = _memory_node_ids(inputs_by_task_id[task_id])
        seen_node_ids: set[str] = set()
        previous_score: float | None = None
        for ranked_node in ranked_nodes:
            if not isinstance(ranked_node, dict):
                raise ContractValidationError(f"Invalid ranked results: task_id={task_id} ranked node is not an object.")
            _reject_unknown_fields(ranked_node, RANKED_NODE_FIELDS, "ranked node", task_id)
            node_id = _required_string(ranked_node, "node_id", "ranked node", task_id)
            if node_id in seen_node_ids:
                raise ContractValidationError(
                    f"Invalid ranked results: task_id={task_id} method={method} ranked_nodes contains duplicate node_id={node_id}."
                )
            seen_node_ids.add(node_id)
            if node_id not in expected_node_ids:
                raise ContractValidationError(
                    f"Invalid ranked results: task_id={task_id} method={method} ranked node_id={node_id} does not exist."
                )
            score = _required_finite_number(ranked_node, "score", "ranked node", task_id)
            if previous_score is not None and score > previous_score:
                raise ContractValidationError(
                    f"Invalid ranked results: task_id={task_id} method={method} ranked_nodes must be sorted descending."
                )
            previous_score = score

        if seen_node_ids != expected_node_ids:
            missing = sorted(expected_node_ids - seen_node_ids)
            extra = sorted(seen_node_ids - expected_node_ids)
            raise ContractValidationError(
                f"Invalid ranked results: task_id={task_id} method={method} ranking must include every memory node exactly once; missing={missing} extra={extra}."
            )

        _required_finite_number(prediction, "latency_ms", "ranked result", task_id, minimum=0.0)
        _required_int(prediction, "input_tokens", "ranked result", task_id, minimum=0)
        _validate_retrieved_subgraph(prediction.get("retrieved_subgraph"), expected_node_ids, task_id)


def _validate_retrieved_subgraph(value: Any, valid_node_ids: set[str], task_id: str) -> None:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} retrieved_subgraph must be an object.")
    _reject_unknown_fields(value, RETRIEVED_SUBGRAPH_FIELDS, "retrieved subgraph", task_id)
    nodes = value.get("nodes")
    edges = value.get("edges")
    if not isinstance(nodes, list):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} retrieved_subgraph.nodes must be a list.")
    if not isinstance(edges, list):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} retrieved_subgraph.edges must be a list.")
    subgraph_node_ids = set(nodes)
    for node_id in subgraph_node_ids:
        if node_id not in valid_node_ids and node_id != "q":
            raise ContractValidationError(
                f"Invalid ranked results: task_id={task_id} retrieved_subgraph node_id={node_id} does not exist."
            )
    for edge in edges:
        edge_node_ids = subgraph_node_ids | {"q"}
        _validate_graph_edge(edge, edge_node_ids, task_id)


__all__ = ["validate_ranked_results"]
