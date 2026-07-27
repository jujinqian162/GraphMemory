from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from pydantic import ValidationError as PydanticValidationError

from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.contracts import ExecutionProvenanceTrace
from graph_memory.retrieval.methods.epgm.config import (
    DEFAULT_RELATION_DESCRIPTIONS,
    EPGM_VARIANTS,
)
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
    if trace_kind == "typed_local_bridge":
        _validate_local_intervention_trace(
            trace,
            valid_candidate_ids,
            task_id,
            proposal_field="bridges",
            endpoint_fields=("source_candidate_id", "target_candidate_id"),
            extra_fields={
                "linked_entity_ids",
                "mentions",
                "title_groups",
                "resolver_evidence",
            },
        )
        return
    if trace_kind == "execution_provenance_local":
        _validate_local_intervention_trace(
            trace,
            valid_candidate_ids,
            task_id,
            proposal_field="paths",
            endpoint_fields=("anchor_id", "partner_id"),
            extra_fields={"edges", "scorer_identity", "variant"},
        )
        return
    if trace_kind == "execution_provenance_subgraph":
        _validate_query_conditioned_subgraph_trace(
            trace, valid_candidate_ids, task_id
        )
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


def _format_pydantic_error(error: PydanticValidationError) -> str:
    first = error.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "<root>"
    return f"{location}: {first['msg']}"


def _validate_execution_provenance_trace(trace: dict[str, Any], task_id: str) -> None:
    try:
        ExecutionProvenanceTrace.model_validate(trace)
    except PydanticValidationError as error:
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {_format_pydantic_error(error)}."
        ) from error


def _validate_local_intervention_trace(
    trace: dict[str, Any],
    valid_candidate_ids: set[str],
    task_id: str,
    *,
    proposal_field: str,
    endpoint_fields: tuple[str, str],
    extra_fields: set[str],
) -> None:
    _reject_unknown_fields(
        trace,
        {
            "trace_kind",
            "dense_ranks",
            "seed_candidate_ids",
            proposal_field,
            "protected_prefix",
            "exact_dense_fallback",
            "emitted_edges",
            *extra_fields,
        },
        "native trace",
        task_id,
    )
    dense_records = _native_trace_records(
        trace.get("dense_ranks"), "dense_ranks", task_id
    )
    dense_ids: list[str] = []
    for record in dense_records:
        _reject_unknown_fields(
            record,
            {"node_id", "dense_rank", "dense_score", "final_rank"},
            "native trace dense rank",
            task_id,
        )
        dense_ids.append(
            _required_string(record, "node_id", "native trace dense rank", task_id)
        )
        _required_int(
            record, "dense_rank", "native trace dense rank", task_id, minimum=1
        )
        _required_int(
            record, "final_rank", "native trace dense rank", task_id, minimum=1
        )
        _required_finite_number(
            record, "dense_score", "native trace dense rank", task_id
        )
    _validate_candidate_references(
        dense_ids, valid_candidate_ids, task_id, "dense_ranks"
    )
    seeds = _native_trace_string_list(
        trace.get("seed_candidate_ids"), "seed_candidate_ids", task_id
    )
    protected = _native_trace_string_list(
        trace.get("protected_prefix"), "protected_prefix", task_id
    )
    _validate_candidate_references(
        [*seeds, *protected], valid_candidate_ids, task_id, "seed/prefix"
    )
    exact_fallback = trace.get("exact_dense_fallback")
    if not isinstance(exact_fallback, bool):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} exact_dense_fallback must be boolean."
        )
    proposals = _native_trace_records(
        trace.get(proposal_field), proposal_field, task_id
    )
    accepted_count = 0
    for proposal in proposals:
        endpoints = [
            _required_string(proposal, field, "native trace proposal", task_id)
            for field in endpoint_fields
        ]
        _validate_candidate_references(
            endpoints, valid_candidate_ids, task_id, "proposal endpoint"
        )
        accepted = proposal.get("accepted")
        if not isinstance(accepted, bool):
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} proposal accepted must be boolean."
            )
        accepted_count += int(accepted)
        reason = proposal.get("rejection_reason")
        if accepted and reason is not None:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} accepted proposal has a rejection reason."
            )
        if not accepted and (not isinstance(reason, str) or not reason):
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} rejected proposal needs a reason."
            )
    if exact_fallback != (accepted_count == 0):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} exact fallback state is inconsistent."
        )
    emitted = _native_trace_records(
        trace.get("emitted_edges"), "emitted_edges", task_id
    )
    for edge in emitted:
        _reject_unknown_fields(
            edge,
            {"source", "target", "edge_type", "confidence"},
            "native trace emitted edge",
            task_id,
        )
        endpoints = [
            _required_string(edge, field, "native trace emitted edge", task_id)
            for field in ("source", "target")
        ]
        _validate_candidate_references(
            endpoints, valid_candidate_ids, task_id, "emitted edge"
        )
        _required_string(edge, "edge_type", "native trace emitted edge", task_id)
        _required_finite_number(
            edge,
            "confidence",
            "native trace emitted edge",
            task_id,
            minimum=0.0,
        )
    if "linked_entity_ids" in extra_fields:
        _native_trace_string_list(
            trace.get("linked_entity_ids"), "linked_entity_ids", task_id
        )
    for field in extra_fields - {"scorer_identity", "linked_entity_ids", "variant"}:
        _native_trace_records(trace.get(field), field, task_id)
    if "scorer_identity" in extra_fields:
        _required_string(trace, "scorer_identity", "native trace", task_id)
    if "variant" in extra_fields:
        variant = _required_string(trace, "variant", "native trace", task_id)
        if variant not in EPGM_VARIANTS:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} unsupported variant={variant}."
            )


def _validate_query_conditioned_subgraph_trace(
    trace: dict[str, Any],
    valid_candidate_ids: set[str],
    task_id: str,
) -> None:
    fields = {
        "trace_kind",
        "native_graph_node_ids",
        "dense_ranks",
        "relation_description_version",
        "relations",
        "transitions",
        "ppr_nodes",
        "ppr_iterations",
        "ppr_residual",
        "ppr_converged",
        "candidate_prizes",
        "selected_candidate_ids",
        "connector_node_ids",
        "selection_steps",
        "selected_native_edges",
        "objective",
        "top_k",
        "exact_dense_fallback",
        "emitted_edges",
        "scorer_identity",
        "variant",
    }
    _reject_unknown_fields(trace, fields, "native trace", task_id)
    native_ids_list = _native_trace_string_list(
        trace.get("native_graph_node_ids"), "native_graph_node_ids", task_id
    )
    if len(native_ids_list) != len(set(native_ids_list)):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} duplicate native graph node ids."
        )
    native_ids = set(native_ids_list)
    if not valid_candidate_ids.issubset(native_ids):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} candidate missing from native graph."
        )

    dense = _native_trace_records(trace.get("dense_ranks"), "dense_ranks", task_id)
    dense_ids: list[str] = []
    for record in dense:
        _reject_unknown_fields(
            record,
            {"node_id", "dense_rank", "dense_score", "final_rank"},
            "native trace dense rank",
            task_id,
        )
        dense_ids.append(
            _required_string(record, "node_id", "native trace dense rank", task_id)
        )
        _required_int(record, "dense_rank", "native trace dense rank", task_id, minimum=1)
        _required_int(record, "final_rank", "native trace dense rank", task_id, minimum=1)
        _required_finite_number(record, "dense_score", "native trace dense rank", task_id)
    if set(dense_ids) != valid_candidate_ids or len(dense_ids) != len(set(dense_ids)):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} dense ranks do not cover candidates exactly."
        )

    _required_string(trace, "relation_description_version", "native trace", task_id)
    relations = _native_trace_records(trace.get("relations"), "relations", task_id)
    relation_types: set[str] = set()
    relation_mass = 0.0
    for record in relations:
        _reject_unknown_fields(
            record,
            {"edge_type", "similarity", "affinity"},
            "native trace relation",
            task_id,
        )
        edge_type = _required_string(record, "edge_type", "native trace relation", task_id)
        if edge_type not in DEFAULT_RELATION_DESCRIPTIONS:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} unknown relation={edge_type}."
            )
        if edge_type in relation_types:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate relation={edge_type}."
            )
        relation_types.add(edge_type)
        _required_finite_number(record, "similarity", "native trace relation", task_id)
        relation_mass += _required_finite_number(
            record, "affinity", "native trace relation", task_id, minimum=0.0
        )
    if relations and not math.isclose(relation_mass, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} relation affinities do not sum to one."
        )

    transitions = _native_trace_records(
        trace.get("transitions"), "transitions", task_id
    )
    transition_keys: set[tuple[str, str, str, str]] = set()
    row_mass: dict[str, float] = defaultdict(float)
    for record in transitions:
        _reject_unknown_fields(
            record,
            {
                "source",
                "target",
                "edge_type",
                "direction",
                "recorded_weight",
                "relation_affinity",
                "probability",
                "cost",
            },
            "native trace transition",
            task_id,
        )
        source = _required_string(record, "source", "native trace transition", task_id)
        target = _required_string(record, "target", "native trace transition", task_id)
        edge_type = _required_string(record, "edge_type", "native trace transition", task_id)
        direction = _required_string(record, "direction", "native trace transition", task_id)
        if source not in native_ids or target not in native_ids or source == target:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} invalid transition endpoint."
            )
        if edge_type not in relation_types or direction not in {"forward", "reverse"}:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} invalid typed transition."
            )
        key = (source, target, edge_type, direction)
        if key in transition_keys:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate transition={key}."
            )
        transition_keys.add(key)
        _required_finite_number(record, "recorded_weight", "native trace transition", task_id, minimum=0.0)
        _required_finite_number(record, "relation_affinity", "native trace transition", task_id, minimum=0.0)
        probability = _required_finite_number(
            record, "probability", "native trace transition", task_id, minimum=0.0
        )
        _required_finite_number(record, "cost", "native trace transition", task_id, minimum=0.0)
        row_mass[source] += probability
    for source, total in row_mass.items():
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} transition row source={source} does not sum to one."
            )

    ppr_nodes = _native_trace_records(trace.get("ppr_nodes"), "ppr_nodes", task_id)
    ppr_ids: set[str] = set()
    teleport_mass = 0.0
    ppr_mass = 0.0
    for record in ppr_nodes:
        _reject_unknown_fields(
            record, {"node_id", "teleport", "score"}, "native trace PPR node", task_id
        )
        node_id = _required_string(record, "node_id", "native trace PPR node", task_id)
        if node_id in ppr_ids:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate PPR node={node_id}."
            )
        ppr_ids.add(node_id)
        teleport_mass += _required_finite_number(
            record, "teleport", "native trace PPR node", task_id, minimum=0.0
        )
        ppr_mass += _required_finite_number(
            record, "score", "native trace PPR node", task_id, minimum=0.0
        )
    if ppr_ids != native_ids or not math.isclose(teleport_mass, 1.0, abs_tol=1e-8) or not math.isclose(ppr_mass, 1.0, abs_tol=1e-8):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} invalid PPR node coverage or mass."
        )
    _required_int(trace, "ppr_iterations", "native trace", task_id, minimum=1)
    _required_finite_number(trace, "ppr_residual", "native trace", task_id, minimum=0.0)
    if not isinstance(trace.get("ppr_converged"), bool):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} ppr_converged must be boolean."
        )

    prizes = _native_trace_records(
        trace.get("candidate_prizes"), "candidate_prizes", task_id
    )
    prize_ids: set[str] = set()
    for record in prizes:
        _reject_unknown_fields(
            record,
            {"node_id", "dense_component", "ppr_component", "prize"},
            "native trace candidate prize",
            task_id,
        )
        node_id = _required_string(record, "node_id", "native trace candidate prize", task_id)
        prize_ids.add(node_id)
        for field in ("dense_component", "ppr_component", "prize"):
            _required_finite_number(record, field, "native trace candidate prize", task_id, minimum=0.0)
    if prize_ids != valid_candidate_ids or len(prizes) != len(prize_ids):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} candidate prizes do not cover candidates exactly."
        )

    selected = _native_trace_string_list(
        trace.get("selected_candidate_ids"), "selected_candidate_ids", task_id
    )
    connectors = _native_trace_string_list(
        trace.get("connector_node_ids"), "connector_node_ids", task_id
    )
    if len(selected) != len(set(selected)) or len(connectors) != len(set(connectors)):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} duplicate selection node."
        )
    _validate_candidate_references(selected, valid_candidate_ids, task_id, "selected candidates")
    if set(connectors) - native_ids or set(connectors) & valid_candidate_ids:
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} invalid connector reference."
        )
    top_k = _required_int(trace, "top_k", "native trace", task_id, minimum=1)
    if len(selected) > top_k:
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} selected candidates exceed top_k."
        )

    steps = _native_trace_records(
        trace.get("selection_steps"), "selection_steps", task_id
    )
    selected_stored_edge_keys: set[tuple[str, str, str]] = set()
    for record in steps:
        _reject_unknown_fields(
            record,
            {
                "anchor_id",
                "target_id",
                "path_node_ids",
                "transitions",
                "added_candidate_ids",
                "displaced_candidate_ids",
                "prize_gain",
                "edge_cost",
                "displacement_cost",
                "marginal_gain",
            },
            "native trace selection step",
            task_id,
        )
        anchor = _required_string(
            record, "anchor_id", "native trace selection step", task_id
        )
        target = _required_string(
            record, "target_id", "native trace selection step", task_id
        )
        path_ids = _native_trace_string_list(
            record.get("path_node_ids"), "path_node_ids", task_id
        )
        if (
            not path_ids
            or set(path_ids) - native_ids
            or path_ids[0] != anchor
            or path_ids[-1] != target
        ):
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} invalid selection path."
            )
        selected_arcs = _native_trace_records(
            record.get("transitions"), "selection transitions", task_id
        )
        if len(selected_arcs) != len(path_ids) - 1:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} selection transition count mismatch."
            )
        for index, arc in enumerate(selected_arcs):
            _reject_unknown_fields(
                arc,
                {"source", "target", "edge_type", "direction"},
                "native trace selected transition",
                task_id,
            )
            source = _required_string(
                arc, "source", "native trace selected transition", task_id
            )
            target = _required_string(
                arc, "target", "native trace selected transition", task_id
            )
            edge_type = _required_string(
                arc, "edge_type", "native trace selected transition", task_id
            )
            direction = _required_string(
                arc, "direction", "native trace selected transition", task_id
            )
            if (
                source != path_ids[index]
                or target != path_ids[index + 1]
                or (source, target, edge_type, direction) not in transition_keys
            ):
                raise ContractValidationError(
                    f"Invalid native trace: task_id={task_id} selected transition is not a traced path arc."
                )
            selected_stored_edge_keys.add(
                (source, target, edge_type)
                if direction == "forward"
                else (target, source, edge_type)
            )
        added = _native_trace_string_list(
            record.get("added_candidate_ids"), "added_candidate_ids", task_id
        )
        _validate_candidate_references(
            added, valid_candidate_ids, task_id, "added candidates"
        )
        if len(added) != len(set(added)) or not set(added).issubset(selected):
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} invalid added candidates."
            )
        displaced = _native_trace_string_list(
            record.get("displaced_candidate_ids"),
            "displaced_candidate_ids",
            task_id,
        )
        _validate_candidate_references(
            displaced, valid_candidate_ids, task_id, "displaced candidates"
        )
        if len(displaced) != len(set(displaced)) or set(displaced) & set(added):
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} invalid displaced candidates."
            )
        for field in ("prize_gain", "edge_cost", "displacement_cost"):
            _required_finite_number(
                record,
                field,
                "native trace selection step",
                task_id,
                minimum=0.0,
            )
        _required_finite_number(
            record, "marginal_gain", "native trace selection step", task_id
        )

    selected_edges = _native_trace_records(
        trace.get("selected_native_edges"), "selected_native_edges", task_id
    )
    adjacency: dict[str, set[str]] = defaultdict(set)
    selected_edge_keys: set[tuple[str, str, str]] = set()
    selected_context = set(selected) | set(connectors)
    for record in selected_edges:
        _reject_unknown_fields(
            record,
            {
                "source",
                "target",
                "edge_type",
                "weight",
                "binding",
                "semantic_rank",
                "semantic_score",
            },
            "native trace selected edge",
            task_id,
        )
        source = _required_string(record, "source", "native trace selected edge", task_id)
        target = _required_string(record, "target", "native trace selected edge", task_id)
        edge_type = _required_string(record, "edge_type", "native trace selected edge", task_id)
        if source not in selected_context or target not in selected_context:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} selected edge leaves selected context."
            )
        key = (source, target, edge_type)
        if key in selected_edge_keys:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} duplicate selected edge={key}."
            )
        selected_edge_keys.add(key)
        adjacency[source].add(target)
        adjacency[target].add(source)
        _required_finite_number(
            record, "weight", "native trace selected edge", task_id, minimum=0.0
        )
        binding = record.get("binding")
        if edge_type == "feeds" and binding is None:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} selected feeds edge needs binding."
            )
        if edge_type != "feeds" and binding is not None:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} binding is only valid on feeds."
            )
        if binding is not None:
            _validate_native_trace_binding(binding, task_id)
        if record.get("semantic_rank") is not None:
            _required_int(
                record,
                "semantic_rank",
                "native trace selected edge",
                task_id,
                minimum=1,
            )
        if record.get("semantic_score") is not None:
            _required_finite_number(
                record, "semantic_score", "native trace selected edge", task_id
            )
    if selected_context:
        seen = {next(iter(selected_context))}
        queue = deque(seen)
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        if seen != selected_context:
            raise ContractValidationError(
                f"Invalid native trace: task_id={task_id} selected subgraph is disconnected."
            )
    if selected_edge_keys != selected_stored_edge_keys:
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} selected native edge orientation is inconsistent."
        )

    emitted = _native_trace_records(trace.get("emitted_edges"), "emitted_edges", task_id)
    for record in emitted:
        _reject_unknown_fields(
            record,
            {"source", "target", "edge_type", "confidence"},
            "native trace emitted edge",
            task_id,
        )
        endpoints = [
            _required_string(record, field, "native trace emitted edge", task_id)
            for field in ("source", "target")
        ]
        _validate_candidate_references(endpoints, valid_candidate_ids, task_id, "emitted edge")
        _required_string(record, "edge_type", "native trace emitted edge", task_id)
        _required_finite_number(record, "confidence", "native trace emitted edge", task_id, minimum=0.0)
    fallback = trace.get("exact_dense_fallback")
    if not isinstance(fallback, bool):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} exact_dense_fallback must be boolean."
        )
    if fallback != (not steps):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} fallback/intervention state is inconsistent."
        )
    if fallback and (selected or connectors or steps or selected_edges):
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} fallback retains a selected subgraph."
        )
    _required_finite_number(trace, "objective", "native trace", task_id)
    _required_string(trace, "scorer_identity", "native trace", task_id)
    variant = _required_string(trace, "variant", "native trace", task_id)
    if variant != "ppr_steiner":
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} unsupported subgraph variant={variant}."
        )


def _validate_candidate_references(
    candidate_ids: Sequence[str],
    valid_candidate_ids: set[str],
    task_id: str,
    field_name: str,
) -> None:
    unknown = sorted(set(candidate_ids) - valid_candidate_ids)
    if unknown:
        raise ContractValidationError(
            f"Invalid native trace: task_id={task_id} {field_name} references "
            f"unknown candidates={unknown}."
        )


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
