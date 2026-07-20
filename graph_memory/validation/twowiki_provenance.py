from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import cast

from graph_memory.datasets.twowiki_provenance.projectors import (
    provenance_graph_from_record,
)
from graph_memory.datasets.twowiki_provenance.records import (
    ProvenanceGraphRecord,
    TWOWIKI_PROVENANCE_SCHEMA_VERSION,
)
from graph_memory.graphs.provenance import (
    ExecutionProvenanceGraph,
    ProvenanceEdgeType,
    ProvenanceNodeType,
    binding_matches_endpoints,
)
from graph_memory.validation.common import (
    ContractValidationError,
    _reject_unknown_fields,
    _require_record_list,
    _require_record_map,
    _require_unique,
    _required_int,
    _required_string,
    validate_no_label_fields,
    validate_task_id_alignment,
)

RANKING_FIELDS = {
    "task_id",
    "question",
    "question_type",
    "candidates",
    "graph",
    "metadata",
}
CANDIDATE_FIELDS = {
    "output_id",
    "call_id",
    "title",
    "sentence_index",
    "position",
    "text",
}
LABEL_FIELDS = {
    "task_id",
    "gold_answer",
    "gold_evidence_output_ids",
    "gold_dependency_edges",
    "metadata",
}
FORBIDDEN_RANKING_KEYS = {
    "gold",
    "is_gold",
    "gold_answer",
    "gold_evidence_output_ids",
    "gold_dependency_edges",
    "supporting_facts",
    "answer",
}
FEED_CONFIDENCE_FIELDS = {
    "synthetic",
    "semantic_scorer",
    "scorer_identity",
    "query_template_version",
    "semantic_rank",
    "semantic_score",
    "normalized_score",
    "source_probability",
    "calibrated_weight",
    "branch_role",
    "rank_bucket",
}


def validate_twowiki_provenance_ranking_records(records: object) -> None:
    values = _require_record_list(records, "2Wiki provenance ranking records")
    seen_task_ids: set[str] = set()
    for value in values:
        _validate_ranking_record(value, seen_task_ids=seen_task_ids)


def _validate_ranking_record(
    value: object, *, seen_task_ids: set[str]
) -> tuple[str, ExecutionProvenanceGraph]:
    if not isinstance(value, dict):
        raise ContractValidationError(
            "Invalid 2Wiki provenance ranking record: record must be an object."
        )
    task_id = _required_string(value, "task_id", "2Wiki provenance ranking record")
    _reject_unknown_fields(
        value, RANKING_FIELDS, "2Wiki provenance ranking record", task_id
    )
    validate_no_label_fields(
        value,
        artifact_name="2Wiki provenance ranking record",
        task_id=task_id,
    )
    _reject_forbidden_keys(value, task_id=task_id)
    _require_unique(task_id, seen_task_ids, "2Wiki provenance ranking task_id")
    if not task_id.startswith("2wiki_provenance_"):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance ranking record: task_id={task_id} prefix."
        )
    _required_string(value, "question", "2Wiki provenance ranking record", task_id)
    _required_string(
        value, "question_type", "2Wiki provenance ranking record", task_id
    )
    metadata = value.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset") != "twowiki_provenance"
        or metadata.get("synthetic_execution_graph") is not True
        or metadata.get("schema_version") != TWOWIKI_PROVENANCE_SCHEMA_VERSION
    ):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance ranking record: task_id={task_id} metadata."
        )
    construction = _validate_construction_identity(metadata, task_id=task_id)
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 4:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance ranking record: task_id={task_id} "
            "requires at least four candidates."
        )
    candidate_pairs = _validate_candidates(candidates, task_id=task_id)
    graph_record = value.get("graph")
    if not isinstance(graph_record, dict):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance ranking record: task_id={task_id} graph."
        )
    try:
        graph = provenance_graph_from_record(
            cast(ProvenanceGraphRecord, cast(object, graph_record))
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id}: {error}"
        ) from error
    if graph.task_id != task_id:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} mismatch."
        )
    node_by_id = {node.node_id: node for node in graph.nodes}
    if {
        node.node_id
        for node in graph.nodes
        if node.node_type is ProvenanceNodeType.TOOL_OUTPUT
    } != set(candidate_pairs):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} candidate outputs mismatch."
        )
    for output_id, call_id in candidate_pairs.items():
        if (
            node_by_id.get(call_id) is None
            or node_by_id[call_id].node_type is not ProvenanceNodeType.TOOL_CALL
        ):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} missing call={call_id}."
            )
        if not any(
            edge.source == call_id
            and edge.target == output_id
            and edge.edge_type is ProvenanceEdgeType.RETURNS
            for edge in graph.edges
        ):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} missing returns pair."
            )
    feed_out = Counter(
        edge.source
        for edge in graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
    )
    if set(feed_out) != set(candidate_pairs) or set(feed_out.values()) != {2}:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} output branch degree mismatch."
        )
    feeds_by_source: defaultdict[str, list[object]] = defaultdict(list)
    for edge in graph.edges:
        if edge.edge_type is ProvenanceEdgeType.FEEDS:
            feeds_by_source[edge.source].append(edge)
        elif edge.weight != 1.0:
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} "
                "non-feed edge weight must be 1.0."
            )
    _validate_feed_confidence(
        feeds_by_source,
        construction=construction,
        task_id=task_id,
    )
    inconsistent_bindings = [
        f"{edge.source}->{edge.target}"
        for edge in graph.edges
        if edge.edge_type is ProvenanceEdgeType.FEEDS
        and not binding_matches_endpoints(edge, node_by_id)
    ]
    if inconsistent_bindings:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} "
            f"inconsistent bindings={sorted(inconsistent_bindings)}."
        )
    return task_id, graph


def validate_twowiki_provenance_label_records(
    labels: object,
    records_by_task_id: object,
) -> None:
    values = _require_record_list(labels, "2Wiki provenance label records")
    rankings = _require_record_map(
        records_by_task_id, "2Wiki provenance ranking records by task_id"
    )
    seen_task_ids: set[str] = set()
    for value in values:
        _validate_label_record(value, rankings=rankings, seen_task_ids=seen_task_ids)
    validate_task_id_alignment(
        "2Wiki provenance ranking/label join", set(rankings), seen_task_ids
    )


def _validate_label_record(
    value: object,
    *,
    rankings: Mapping[str, object],
    seen_task_ids: set[str],
    graph: ExecutionProvenanceGraph | None = None,
) -> None:
    if not isinstance(value, dict):
        raise ContractValidationError(
            "Invalid 2Wiki provenance label record: record must be an object."
        )
    task_id = _required_string(value, "task_id", "2Wiki provenance label record")
    _reject_unknown_fields(
        value, LABEL_FIELDS, "2Wiki provenance label record", task_id
    )
    _require_unique(task_id, seen_task_ids, "2Wiki provenance label task_id")
    ranking = rankings.get(task_id)
    if not isinstance(ranking, dict):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance label: task_id={task_id} missing ranking."
        )
    _required_string(value, "gold_answer", "2Wiki provenance label record", task_id)
    gold_ids = value.get("gold_evidence_output_ids")
    edges = value.get("gold_dependency_edges")
    if (
        not isinstance(gold_ids, list)
        or len(gold_ids) != 2
        or len(set(gold_ids)) != 2
        or not all(isinstance(item, str) and item for item in gold_ids)
    ):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance label: task_id={task_id} gold outputs."
        )
    if not isinstance(edges, list) or edges != [[gold_ids[0], gold_ids[1]]]:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance label: task_id={task_id} ordered edge."
        )
    candidates = ranking.get("candidates")
    candidate_by_output = (
        {
            item.get("output_id"): item
            for item in candidates
            if isinstance(candidates, list) and isinstance(item, dict)
        }
        if isinstance(candidates, list)
        else {}
    )
    if any(item not in candidate_by_output for item in gold_ids):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance label: task_id={task_id} gold missing."
        )
    if graph is None:
        graph_record = ranking.get("graph")
        if not isinstance(graph_record, dict):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance label: task_id={task_id} graph missing."
            )
        graph = provenance_graph_from_record(
            cast(ProvenanceGraphRecord, cast(object, graph_record))
        )
    target_call = candidate_by_output[gold_ids[1]].get("call_id")
    has_feeds = any(
        edge.source == gold_ids[0]
        and edge.target == target_call
        and edge.edge_type is ProvenanceEdgeType.FEEDS
        and edge.binding is not None
        for edge in graph.edges
    )
    has_returns = any(
        edge.source == target_call
        and edge.target == gold_ids[1]
        and edge.edge_type is ProvenanceEdgeType.RETURNS
        for edge in graph.edges
    )
    if not has_feeds or not has_returns:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance label: task_id={task_id} gold path missing."
        )


def validate_twowiki_provenance_record(ranking: object, label: object) -> None:
    """Validate one ranking+label pair while rebuilding the graph only once.

    Equivalent to calling the ranking and label validators in turn, but the
    execution graph is materialized a single time and shared between the two
    checks. The per-record cost dominates prepare-split on the full 2Wiki set.
    """
    task_id, graph = _validate_ranking_record(ranking, seen_task_ids=set())
    _validate_label_record(
        label,
        rankings={task_id: ranking},
        seen_task_ids=set(),
        graph=graph,
    )


def _validate_candidates(candidates: list[object], *, task_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    seen_calls: set[str] = set()
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance candidate: task_id={task_id} object."
            )
        _reject_unknown_fields(
            candidate, CANDIDATE_FIELDS, "2Wiki provenance candidate", task_id
        )
        output_id = _required_string(
            candidate, "output_id", "2Wiki provenance candidate", task_id
        )
        call_id = _required_string(
            candidate, "call_id", "2Wiki provenance candidate", task_id
        )
        _require_unique(output_id, set(result), f"candidate output task_id={task_id}")
        _require_unique(call_id, seen_calls, f"candidate call task_id={task_id}")
        _required_string(candidate, "title", "2Wiki provenance candidate", task_id)
        _required_string(candidate, "text", "2Wiki provenance candidate", task_id)
        _required_int(
            candidate,
            "sentence_index",
            "2Wiki provenance candidate",
            task_id,
            minimum=0,
        )
        actual_position = _required_int(
            candidate,
            "position",
            "2Wiki provenance candidate",
            task_id,
            minimum=0,
        )
        if actual_position != position:
            raise ContractValidationError(
                f"Invalid 2Wiki provenance candidate: task_id={task_id} position."
            )
        result[output_id] = call_id
    return result


def _reject_forbidden_keys(value: object, *, task_id: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(
            key
            for key in value
            if isinstance(key, str) and key.casefold() in FORBIDDEN_RANKING_KEYS
        )
        if forbidden:
            raise ContractValidationError(
                f"Invalid 2Wiki provenance ranking record: task_id={task_id} forbidden fields={forbidden}."
            )
        for item in value.values():
            _reject_forbidden_keys(item, task_id=task_id)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item, task_id=task_id)


def _validate_construction_identity(
    metadata: Mapping[str, object],
    *,
    task_id: str,
) -> dict[str, object]:
    construction = metadata.get("graph_construction")
    required = {
        "strategy",
        "successors_per_output",
        "hybrid_dense_weight",
        "scorer_identity",
        "query_template_version",
        "semantic_temperature",
        "weight_floor",
        "branch_policy_version",
        "rank_buckets",
    }
    if not isinstance(construction, dict) or set(construction) != required:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} "
            "schema/config mismatch."
        )
    if construction.get("successors_per_output") != 2:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} fixed out-degree config."
        )
    floor = construction.get("weight_floor")
    temperature = construction.get("semantic_temperature")
    if (
        not isinstance(floor, int | float)
        or isinstance(floor, bool)
        or not 0.0 <= float(floor) < 1.0
        or not isinstance(temperature, int | float)
        or isinstance(temperature, bool)
        or float(temperature) <= 0.0
    ):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} calibration config."
        )
    buckets = construction.get("rank_buckets")
    if not isinstance(buckets, dict) or set(buckets) != {"near", "mid", "tail"}:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} rank buckets."
        )
    for name, bounds in buckets.items():
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not isinstance(bounds[0], int)
            or isinstance(bounds[0], bool)
            or (bounds[1] is not None and not isinstance(bounds[1], int))
        ):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} "
                f"rank bucket={name}."
            )
    return cast(dict[str, object], construction)


def _validate_feed_confidence(
    feeds_by_source: Mapping[str, list[object]],
    *,
    construction: Mapping[str, object],
    task_id: str,
) -> None:
    floor = float(cast(float, construction["weight_floor"]))
    expected_mass = 2.0 * floor + (1.0 - floor)
    buckets = cast(dict[str, list[int | None]], construction["rank_buckets"])
    for source, raw_edges in feeds_by_source.items():
        edges = [cast(object, edge) for edge in raw_edges]
        roles: Counter[str] = Counter()
        probability_sum = 0.0
        weight_sum = 0.0
        for raw_edge in edges:
            edge = cast(object, raw_edge)
            metadata = getattr(edge, "metadata")
            weight = float(getattr(edge, "weight"))
            if not isinstance(metadata, Mapping) or set(metadata) != FEED_CONFIDENCE_FIELDS:
                raise ContractValidationError(
                    f"Invalid 2Wiki provenance graph: task_id={task_id} "
                    f"source={source} confidence metadata."
                )
            role = _required_metadata_string(metadata, "branch_role", task_id)
            bucket = _required_metadata_string(metadata, "rank_bucket", task_id)
            semantic_rank = _required_metadata_int(metadata, "semantic_rank", task_id)
            probability = _required_metadata_float(
                metadata, "source_probability", task_id
            )
            calibrated_weight = _required_metadata_float(
                metadata, "calibrated_weight", task_id
            )
            normalized_score = _required_metadata_float(
                metadata, "normalized_score", task_id
            )
            _required_metadata_float(metadata, "semantic_score", task_id)
            if (
                metadata.get("synthetic") is not True
                or metadata.get("semantic_scorer") != construction["strategy"]
                or metadata.get("scorer_identity") != construction["scorer_identity"]
                or metadata.get("query_template_version")
                != construction["query_template_version"]
            ):
                raise ContractValidationError(
                    f"Invalid 2Wiki provenance graph: task_id={task_id} "
                    f"source={source} schema/config mismatch."
                )
            if not 0.0 <= normalized_score <= 1.0 or not 0.0 <= probability <= 1.0:
                raise ContractValidationError(
                    f"Invalid 2Wiki provenance graph: task_id={task_id} "
                    f"source={source} confidence range."
                )
            if (
                not floor <= weight <= 1.0
                or not math.isclose(
                    weight, calibrated_weight, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ContractValidationError(
                    f"Invalid 2Wiki provenance graph: task_id={task_id} "
                    f"source={source} feed weight."
                )
            if role == "semantic_head":
                valid_role = semantic_rank == 1 and bucket == "head"
            elif role == "rank_banded_branch" and bucket in buckets:
                lower, upper = buckets[bucket]
                valid_role = (
                    isinstance(lower, int)
                    and semantic_rank >= lower
                    and (upper is None or semantic_rank <= upper)
                )
            else:
                valid_role = False
            if not valid_role:
                raise ContractValidationError(
                    f"Invalid 2Wiki provenance graph: task_id={task_id} "
                    f"source={source} unmatched rank bucket."
                )
            roles[role] += 1
            probability_sum += probability
            weight_sum += weight
        if roles != {"semantic_head": 1, "rank_banded_branch": 1}:
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} "
                f"source={source} branch roles."
            )
        if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} "
                f"source={source} source probability mass."
            )
        if not math.isclose(weight_sum, expected_mass, rel_tol=0.0, abs_tol=1e-9):
            raise ContractValidationError(
                f"Invalid 2Wiki provenance graph: task_id={task_id} "
                f"source={source} feed mass."
            )


def _required_metadata_string(
    metadata: Mapping[str, object], key: str, task_id: str
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} metadata={key}."
        )
    return value


def _required_metadata_int(
    metadata: Mapping[str, object], key: str, task_id: str
) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} metadata={key}."
        )
    return value


def _required_metadata_float(
    metadata: Mapping[str, object], key: str, task_id: str
) -> float:
    value = metadata.get(key)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ContractValidationError(
            f"Invalid 2Wiki provenance graph: task_id={task_id} metadata={key}."
        )
    return float(value)


__all__ = [
    "validate_twowiki_provenance_label_records",
    "validate_twowiki_provenance_ranking_records",
    "validate_twowiki_provenance_record",
]
