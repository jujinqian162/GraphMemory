from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.graphs.provenance import ProvenanceEdgeType
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.validation.common import (
    ContractValidationError,
    _reject_unknown_fields,
    _require_record_list,
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
RETRIEVAL_METHOD_IDS = {member.value for member in RetrievalMethodId}


def validate_ranked_results(predictions: object, expected_candidates_by_task_id: object) -> None:
    predictions = _require_record_list(predictions, "ranked results")
    expected_by_task_id = _expected_candidate_ids_by_task_id(expected_candidates_by_task_id)

    seen_task_ids: set[str] = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ContractValidationError(f"Invalid ranked results: record index={index} is not an object.")
        task_id = _required_string(prediction, "task_id", "ranked result")
        _reject_unknown_fields(prediction, RANKED_RESULT_FIELDS, "ranked result", task_id)
        _require_unique(task_id, seen_task_ids, "ranked result task_id")
        if task_id not in expected_by_task_id:
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} has no expected candidate ids.")

        method = _required_string(prediction, "method", "ranked result", task_id)
        if method not in RETRIEVAL_METHOD_IDS:
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} unsupported method={method}.")

        ranked_nodes = prediction.get("ranked_nodes")
        if not isinstance(ranked_nodes, list):
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} ranked_nodes must be a list.")
        expected_node_ids = expected_by_task_id[task_id]
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
                f"Invalid ranked results: task_id={task_id} method={method} ranking must include every candidate exactly once; missing={missing} extra={extra}."
            )

        _required_finite_number(prediction, "latency_ms", "ranked result", task_id, minimum=0.0)
        _required_int(prediction, "input_tokens", "ranked result", task_id, minimum=0)
        _validate_metadata(prediction.get("metadata"), expected_node_ids, task_id)
        _validate_retrieved_subgraph(prediction.get("retrieved_subgraph"), expected_node_ids, task_id)


def expected_candidate_ids_from_requests(requests: Sequence[TextRankingRequest]) -> dict[str, set[str]]:
    return {request.task_id: {candidate.item_id for candidate in request.candidates} for request in requests}


def _expected_candidate_ids_by_task_id(value: object) -> dict[str, set[str]]:
    if isinstance(value, Mapping):
        expected: dict[str, set[str]] = {}
        for task_id, raw_ids in value.items():
            if not isinstance(task_id, str) or not task_id:
                raise ContractValidationError("Invalid expected candidate ids: task ids must be non-empty strings.")
            expected[task_id] = _string_id_set(raw_ids, artifact_name="expected candidate ids", task_id=task_id)
        return expected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, TextRankingRequest) for item in value
    ):
        return expected_candidate_ids_from_requests(cast(Sequence[TextRankingRequest], value))
    raise ContractValidationError("Invalid expected candidate ids: expected mapping or TextRankingRequest sequence.")


def _string_id_set(value: object, *, artifact_name: str, task_id: str) -> set[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} ids must be a sequence.")
    ids: set[str] = set()
    for item_id in value:
        if not isinstance(item_id, str) or not item_id:
            raise ContractValidationError(f"Invalid {artifact_name}: task_id={task_id} ids must be non-empty strings.")
        ids.add(item_id)
    return ids


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


def _validate_metadata(value: Any, valid_candidate_ids: set[str], task_id: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} metadata must be an object.")
    if "path_metrics_supported" in value:
        raise ContractValidationError(
            f"Invalid ranked results: task_id={task_id} metadata.path_metrics_supported is not supported; "
            "path metric capability is declared by the method registry."
        )
    if "native_trace" in value:
        _validate_native_trace(value["native_trace"], valid_candidate_ids, task_id)


def _validate_native_trace(value: object, valid_candidate_ids: set[str], task_id: str) -> None:
    trace = _native_trace_record(value, task_id)
    trace_kind = _required_string(trace, "trace_kind", "native trace", task_id)
    if trace_kind == "entity_search":
        _validate_entity_search_trace(trace, valid_candidate_ids, task_id)
        return
    if trace_kind == "execution_provenance":
        _validate_execution_provenance_trace(trace, task_id)
        return
    raise ContractValidationError(
        f"Invalid native trace: task_id={task_id} unsupported trace_kind={trace_kind}."
    )


def _validate_entity_search_trace(
    trace: dict[str, Any], valid_candidate_ids: set[str], task_id: str
) -> None:
    _reject_unknown_fields(
        trace,
        {"trace_kind", "entity_ids", "linked_entity_ids", "seed_entity_ids", "relations"},
        "native trace",
        task_id,
    )
    entity_ids = _native_trace_string_list(trace.get("entity_ids"), "entity_ids", task_id)
    linked_ids = _native_trace_string_list(
        trace.get("linked_entity_ids"), "linked_entity_ids", task_id
    )
    seed_ids = _native_trace_string_list(
        trace.get("seed_entity_ids"), "seed_entity_ids", task_id
    )
    entity_id_set = set(entity_ids)
    for field_name, ids in (("linked_entity_ids", linked_ids), ("seed_entity_ids", seed_ids)):
        unknown = sorted(set(ids) - entity_id_set)
        if unknown:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} {field_name} references unknown entities={unknown}."
            )

    relations = _native_trace_records(trace.get("relations"), "relations", task_id)
    seen_relations: set[tuple[str, str]] = set()
    for relation in relations:
        _reject_unknown_fields(
            relation,
            {"source_entity_id", "target_entity_id", "weight", "candidate_ids"},
            "native trace relation",
            task_id,
        )
        source = _required_string(relation, "source_entity_id", "native trace relation", task_id)
        target = _required_string(relation, "target_entity_id", "native trace relation", task_id)
        if source not in entity_id_set or target not in entity_id_set:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} relation endpoint {source}->{target} is unknown."
            )
        if source == target:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} relation cannot be a self loop."
            )
        relation_key = (min(source, target), max(source, target))
        if relation_key in seen_relations:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate relation={source}->{target}."
            )
        seen_relations.add(relation_key)
        relation_weight = _required_finite_number(
            relation, "weight", "native trace relation", task_id
        )
        if relation_weight <= 0.0:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} relation weight must be positive."
            )
        candidate_ids = _native_trace_string_list(
            relation.get("candidate_ids"),
            "relation.candidate_ids",
            task_id,
            allow_empty=False,
        )
        unknown_candidates = sorted(set(candidate_ids) - valid_candidate_ids)
        if unknown_candidates:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} relation references unknown candidates={unknown_candidates}."
            )


def _validate_execution_provenance_trace(trace: dict[str, Any], task_id: str) -> None:
    _reject_unknown_fields(
        trace,
        {"trace_kind", "node_ids", "paths", "edges"},
        "native trace",
        task_id,
    )
    node_ids = _native_trace_string_list(trace.get("node_ids"), "node_ids", task_id)
    node_id_set = set(node_ids)
    edges = _native_trace_records(trace.get("edges"), "edges", task_id)
    seen_edges: set[tuple[str, str, str]] = set()
    adjacent_pairs: set[frozenset[str]] = set()
    for edge in edges:
        _reject_unknown_fields(
            edge,
            {"source", "target", "edge_type", "weight", "binding"},
            "native trace edge",
            task_id,
        )
        source = _required_string(edge, "source", "native trace edge", task_id)
        target = _required_string(edge, "target", "native trace edge", task_id)
        edge_type = _required_string(edge, "edge_type", "native trace edge", task_id)
        if source not in node_id_set or target not in node_id_set:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} edge endpoint {source}->{target} is unknown."
            )
        if source == target:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} edge cannot be a self loop."
            )
        try:
            typed_edge = ProvenanceEdgeType(edge_type)
        except ValueError as error:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} unsupported edge_type={edge_type}."
            ) from error
        edge_key = (source, target, edge_type)
        if edge_key in seen_edges:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate edge={edge_key}."
            )
        seen_edges.add(edge_key)
        adjacent_pairs.add(frozenset((source, target)))
        _required_finite_number(edge, "weight", "native trace edge", task_id, minimum=0.0)
        binding = edge.get("binding")
        if typed_edge is ProvenanceEdgeType.FEEDS:
            _validate_native_trace_binding(binding, task_id)
        elif binding is not None:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} binding is only valid on feeds edges."
            )

    paths = _native_trace_records(trace.get("paths"), "paths", task_id)
    seen_paths: set[tuple[str, ...]] = set()
    score_fields = {
        "score",
        "semantic_relevance",
        "binding_consistency",
        "provenance_completeness",
        "explicit_grounding",
        "path_length_penalty",
        "invalidation_penalty",
    }
    for path in paths:
        _reject_unknown_fields(
            path,
            {"node_ids", *score_fields},
            "native trace path",
            task_id,
        )
        path_node_ids = _native_trace_string_list(
            path.get("node_ids"), "path.node_ids", task_id, allow_empty=False
        )
        if path_node_ids in seen_paths:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate path={path_node_ids}."
            )
        seen_paths.add(path_node_ids)
        unknown_nodes = sorted(set(path_node_ids) - node_id_set)
        if unknown_nodes:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} path references unknown nodes={unknown_nodes}."
            )
        for source, target in zip(path_node_ids, path_node_ids[1:], strict=False):
            if frozenset((source, target)) not in adjacent_pairs:
                raise ContractValidationError(
                    f"Invalid native trace: task_id={task_id} path step {source}->{target} has no traced edge."
                )
        for field_name in score_fields:
            _required_finite_number(path, field_name, "native trace path", task_id)


def _validate_native_trace_binding(value: object, task_id: str) -> None:
    binding = _native_trace_record(value, task_id, component="binding")
    _reject_unknown_fields(
        binding,
        {"output_field", "input_parameter", "binding_kind"},
        "native trace binding",
        task_id,
    )
    for field_name in ("output_field", "input_parameter", "binding_kind"):
        _required_string(binding, field_name, "native trace binding", task_id)


def _native_trace_record(
    value: object, task_id: str, *, component: str = "trace"
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {component} must be an object."
        )
    return value


def _native_trace_records(value: object, field_name: str, task_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {field_name} must be a list of objects."
        )
    return cast(list[dict[str, Any]], value)


def _native_trace_string_list(
    value: object,
    field_name: str,
    task_id: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {field_name} must be a list of non-empty strings."
        )
    if not allow_empty and not value:
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {field_name} must be non-empty."
        )
    if len(value) != len(set(value)):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {field_name} contains duplicates."
        )
    return tuple(value)

__all__ = ["expected_candidate_ids_from_requests", "validate_ranked_results"]
