from __future__ import annotations

import math
from collections.abc import Mapping

from graph_memory.types import GraphRerankConfig

GRAPH_RERANK_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "lambda_init",
        "lambda_query",
        "lambda_neighbor",
        "lambda_bridge",
        "lambda_path",
        "seed_top_s",
        "max_hops",
        "neighbor_type_weights",
    }
)


def ensure_graph_rerank_config(
    value: GraphRerankConfig | Mapping[str, object] | None,
) -> GraphRerankConfig:
    if value is None:
        raise ValueError("Graph rerank methods require graph_config.")
    if isinstance(value, GraphRerankConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Graph rerank config must be a GraphRerankConfig or mapping.")
    return parse_graph_rerank_config_record(value)


def parse_graph_rerank_config_record(record: Mapping[str, object]) -> GraphRerankConfig:
    if "type_weights" in record:
        raise ValueError("type_weights is deprecated; use neighbor_type_weights instead.")

    unknown_fields = sorted(set(record) - GRAPH_RERANK_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(f"Graph rerank config contains unsupported fields: {unknown_fields}.")

    defaults = GraphRerankConfig()
    return GraphRerankConfig(
        lambda_init=_float_field(record, "lambda_init", defaults.lambda_init),
        lambda_query=_float_field(record, "lambda_query", defaults.lambda_query),
        lambda_neighbor=_float_field(record, "lambda_neighbor", defaults.lambda_neighbor),
        lambda_bridge=_float_field(record, "lambda_bridge", defaults.lambda_bridge),
        lambda_path=_float_field(record, "lambda_path", defaults.lambda_path),
        seed_top_s=_int_field(record, "seed_top_s", defaults.seed_top_s),
        max_hops=_int_field(record, "max_hops", defaults.max_hops),
        neighbor_type_weights=_neighbor_type_weights(record, defaults.neighbor_type_weights),
    )


def _float_field(record: Mapping[str, object], field_name: str, default: float) -> float:
    value = record.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Graph rerank config field {field_name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Graph rerank config field {field_name} must be a finite number.")
    return number


def _int_field(record: Mapping[str, object], field_name: str, default: int) -> int:
    value = record.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Graph rerank config field {field_name} must be an integer.")
    return value


def _neighbor_type_weights(
    record: Mapping[str, object],
    default: Mapping[str, float],
) -> dict[str, float]:
    value = record.get("neighbor_type_weights", default)
    if not isinstance(value, Mapping):
        raise ValueError("Graph rerank config field neighbor_type_weights must be a mapping.")

    weights: dict[str, float] = {}
    for edge_type, raw_weight in value.items():
        if not isinstance(edge_type, str):
            raise ValueError("Graph rerank config neighbor_type_weights keys must be strings.")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(
                f"Graph rerank config neighbor_type_weights[{edge_type}] must be a finite number."
            )
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise ValueError(
                f"Graph rerank config neighbor_type_weights[{edge_type}] must be a finite number."
            )
        weights[edge_type] = weight
    return weights
