from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, TypeAlias, cast

from graph_memory.types import ALLOWED_EDGE_TYPES, ALLOWED_NODE_TYPES, SUPPORTED_METHODS


class ContractValidationError(ValueError):
    """Raised when an artifact violates a documented Phase 1 contract."""


ValidationRecord: TypeAlias = dict[str, Any]
ValidationRecords: TypeAlias = list[ValidationRecord]
ValidationRecordMap: TypeAlias = dict[str, ValidationRecord]


def as_validation_records(records: object) -> ValidationRecords:
    """Return a zero-copy validation view for TypedDict artifact records."""

    return cast(ValidationRecords, records)


def as_validation_record_map(records_by_key: object) -> ValidationRecordMap:
    """Return a zero-copy validation view for maps keyed by task id."""

    return cast(ValidationRecordMap, records_by_key)


FORBIDDEN_LABEL_FIELDS: set[str] = {
    "gold_answer",
    "gold_evidence_nodes",
    "gold_dependency_edges",
    "supporting_facts",
    "is_gold",
    "is_gold_evidence",
    "is_gold_edge",
}

MEMORY_TASK_INPUT_FIELDS = {"task_id", "query", "memory_items", "metadata", "debug"}
MEMORY_ITEM_FIELDS = {"id", "node_type", "text", "source", "sentence_id", "position"}
LABEL_FIELDS = {"task_id", "gold_answer", "gold_evidence_nodes", "gold_dependency_edges", "metadata", "debug"}
GRAPH_FIELDS = {"task_id", "nodes", "edges", "metadata", "debug"}
QUESTION_NODE_FIELDS = {"id", "node_type", "text"}
GRAPH_EDGE_FIELDS = {"source", "target", "edge_type", "weight", "directed"}
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

METRIC_COLUMNS = [
    "Method",
    "Recall@2",
    "Recall@5",
    "Recall@10",
    "Evidence F1@5",
    "Evidence F1@10",
    "Full Support@5",
    "Full Support@10",
    "MRR",
    "Connected Evidence Recall@5",
    "Connected Evidence Recall@10",
    "Query-Evidence Connectivity@10",
    "Path Recall@10",
    "Edge Recall@10",
    "Retrieval Latency / Query",
]

def validate_no_label_fields(value: Any, *, artifact_name: str = "artifact", task_id: str | None = None) -> None:
    _walk_forbidden_fields(value, artifact_name=artifact_name, task_id=task_id, path=artifact_name)


def validate_memory_task_inputs(records: ValidationRecords) -> None:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid memory task inputs: artifact must be a list.")

    seen_task_ids: set[str] = set()
    for index, task_input in enumerate(records):
        if not isinstance(task_input, dict):
            raise ContractValidationError(f"Invalid memory task inputs: record index={index} is not an object.")
        task_id = _required_string(task_input, "task_id", "memory task input")
        _reject_unknown_fields(task_input, MEMORY_TASK_INPUT_FIELDS, "memory task input", task_id)
        validate_no_label_fields(task_input, artifact_name="memory task input", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "memory task input task_id")
        _required_string(task_input, "query", "memory task input", task_id)

        memory_items = task_input.get("memory_items")
        if not isinstance(memory_items, list) or not memory_items:
            raise ContractValidationError(f"Invalid memory task input: task_id={task_id} memory_items must be non-empty.")

        seen_node_ids: set[str] = set()
        for expected_position, memory_item in enumerate(memory_items):
            if not isinstance(memory_item, dict):
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} memory item index={expected_position} is not an object."
                )
            _reject_unknown_fields(memory_item, MEMORY_ITEM_FIELDS, "memory item", task_id)
            node_id = _required_string(memory_item, "id", "memory item", task_id)
            _require_unique(node_id, seen_node_ids, f"memory item id task_id={task_id}")
            if node_id != f"m{expected_position}":
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} expected id=m{expected_position}."
                )
            if memory_item.get("node_type") != "document_sentence":
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} node_type must be document_sentence."
                )
            _required_string(memory_item, "text", "memory item", task_id)
            _required_string(memory_item, "source", "memory item", task_id)
            _required_int(memory_item, "sentence_id", "memory item", task_id, minimum=0)
            position = _required_int(memory_item, "position", "memory item", task_id, minimum=0)
            if position != expected_position:
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} position={position} expected {expected_position}."
                )


def validate_memory_task_labels(records: ValidationRecords, inputs_by_task_id: ValidationRecordMap) -> None:
    if not isinstance(records, list):
        raise ContractValidationError("Invalid memory task labels: artifact must be a list.")

    seen_task_ids: set[str] = set()
    for index, task_labels in enumerate(records):
        if not isinstance(task_labels, dict):
            raise ContractValidationError(f"Invalid memory task labels: record index={index} is not an object.")
        task_id = _required_string(task_labels, "task_id", "memory task labels")
        _reject_unknown_fields(task_labels, LABEL_FIELDS, "memory task labels", task_id)
        _require_unique(task_id, seen_task_ids, "memory task labels task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid memory task labels: task_id={task_id} has no matching input task.")

        _required_string(task_labels, "gold_answer", "memory task labels", task_id)
        gold_nodes = task_labels.get("gold_evidence_nodes")
        if not isinstance(gold_nodes, list) or not gold_nodes:
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_evidence_nodes must be a non-empty list."
            )
        if len(gold_nodes) != len(set(gold_nodes)):
            raise ContractValidationError(f"Invalid memory task labels: task_id={task_id} duplicate gold evidence node.")

        valid_node_ids = _memory_node_ids(inputs_by_task_id[task_id])
        for node_id in gold_nodes:
            if not isinstance(node_id, str) or node_id not in valid_node_ids:
                raise ContractValidationError(
                    f"Invalid memory task labels: task_id={task_id} gold node={node_id} does not exist in input task."
                )

        dependency_edges = task_labels.get("gold_dependency_edges")
        if not isinstance(dependency_edges, list):
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_dependency_edges must be a list."
            )
        if dependency_edges:
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_dependency_edges must be empty for HotpotQA Phase 1."
            )


def validate_graphs(graphs: ValidationRecords, inputs_by_task_id: ValidationRecordMap) -> None:
    if not isinstance(graphs, list):
        raise ContractValidationError("Invalid graphs: artifact must be a list.")

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


def validate_ranked_results(predictions: ValidationRecords, inputs_by_task_id: ValidationRecordMap) -> None:
    if not isinstance(predictions, list):
        raise ContractValidationError("Invalid ranked results: artifact must be a list.")

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
        if method not in SUPPORTED_METHODS:
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


def validate_graph_rerank_config(config: dict | object) -> None:
    config_dict = _to_plain_dict(config)
    lambda_fields = ["lambda_init", "lambda_query", "lambda_neighbor", "lambda_bridge", "lambda_path"]
    for field_name in lambda_fields:
        value = config_dict.get(field_name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ContractValidationError(f"Invalid graph rerank config: {field_name} must be a finite non-negative number.")
    if float(config_dict.get("lambda_path", 0.0)) != 0.0:
        raise ContractValidationError("Invalid graph rerank config: lambda_path must remain 0.0 for HotpotQA Phase 1.")
    for field_name in ["seed_top_s", "max_hops"]:
        value = config_dict.get(field_name)
        if not isinstance(value, int) or value <= 0:
            raise ContractValidationError(f"Invalid graph rerank config: {field_name} must be a positive integer.")
    type_weights = config_dict.get("type_weights")
    if not isinstance(type_weights, dict):
        raise ContractValidationError("Invalid graph rerank config: type_weights must be an object.")
    for edge_type in ALLOWED_EDGE_TYPES:
        if edge_type not in type_weights:
            raise ContractValidationError(f"Invalid graph rerank config: missing type weight for edge_type={edge_type}.")
        value = type_weights[edge_type]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ContractValidationError(
                f"Invalid graph rerank config: type_weights[{edge_type}] must be a finite non-negative number."
            )


def validate_metric_rows(rows: ValidationRecords) -> None:
    if not isinstance(rows, list):
        raise ContractValidationError("Invalid metric rows: artifact must be a list.")
    for row in rows:
        if not isinstance(row, dict):
            raise ContractValidationError("Invalid metric rows: row is not an object.")
        missing = [column for column in METRIC_COLUMNS if column not in row]
        if missing:
            raise ContractValidationError(f"Invalid metric rows: missing columns={missing}.")
        for column in METRIC_COLUMNS:
            if column in {"Method", "Path Recall@10", "Edge Recall@10"}:
                continue
            value = float(row[column])
            if not math.isfinite(value):
                raise ContractValidationError(f"Invalid metric rows: column={column} must be finite.")
            if column == "Retrieval Latency / Query":
                if value < 0.0:
                    raise ContractValidationError("Invalid metric rows: latency must be non-negative.")
            elif value < 0.0 or value > 1.0:
                raise ContractValidationError(f"Invalid metric rows: column={column} must be in [0.0, 1.0].")


def validate_task_id_alignment(artifact_name: str, expected_task_ids: set[str], observed_task_ids: set[str]) -> None:
    if expected_task_ids != observed_task_ids:
        missing = sorted(expected_task_ids - observed_task_ids)
        extra = sorted(observed_task_ids - expected_task_ids)
        raise ContractValidationError(
            f"Invalid {artifact_name}: task_id alignment mismatch; missing={missing} extra={extra}."
        )


def _validate_graph_nodes(nodes: list, task_input: dict, task_id: str) -> set[str]:
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


def _validate_graph_edge(edge: dict, graph_node_ids: set[str], task_id: str) -> None:
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


def _walk_forbidden_fields(value: Any, *, artifact_name: str, task_id: str | None, path: str) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in FORBIDDEN_LABEL_FIELDS:
                location = f" task_id={task_id}" if task_id is not None else ""
                raise ContractValidationError(
                    f"Invalid {artifact_name}:{location} forbidden label field {key} at {path}.{key}."
                )
            _walk_forbidden_fields(nested_value, artifact_name=artifact_name, task_id=task_id, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _walk_forbidden_fields(nested_value, artifact_name=artifact_name, task_id=task_id, path=f"{path}[{index}]")


def _reject_unknown_fields(record: dict, allowed_fields: set[str], artifact_name: str, task_id: str | None = None) -> None:
    unknown = sorted(set(record) - allowed_fields)
    if unknown:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} unknown fields={unknown}.")


def _required_string(record: dict, field_name: str, artifact_name: str, task_id: str | None = None) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be a non-empty string.")
    return value


def _required_int(
    record: dict,
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
    *,
    minimum: int | None = None,
) -> int:
    value = record.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be an integer.")
    if minimum is not None and value < minimum:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be >= {minimum}.")
    return value


def _required_finite_number(
    record: dict,
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
    *,
    minimum: float | None = None,
) -> float:
    value = record.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be finite.")
    number = float(value)
    if minimum is not None and number < minimum:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be >= {minimum}.")
    return number


def _require_unique(value: str, seen_values: set[str], artifact_name: str) -> None:
    if value in seen_values:
        raise ContractValidationError(f"Invalid {artifact_name}: duplicate value={value}.")
    seen_values.add(value)


def _memory_node_ids(task_input: dict) -> set[str]:
    memory_items = task_input.get("memory_items")
    if not isinstance(memory_items, list):
        return set()
    return {memory_item["id"] for memory_item in memory_items if isinstance(memory_item, dict) and "id" in memory_item}


def _to_plain_dict(config: dict | object) -> dict:
    if isinstance(config, dict):
        return config
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    raise ContractValidationError("Invalid config: expected dict or dataclass instance.")
