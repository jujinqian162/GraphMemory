from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, ensure_graph_rerank_config
from graph_memory.tuning.grid_search import ParameterGrid


def graph_rerank_grid() -> list[GraphRerankConfig]:
    return graph_rerank_grid_from_record(
        {
            "lambda_init": [1.0],
            "lambda_query": [0.0, 0.05, 0.1, 0.2],
            "lambda_neighbor": [0.0, 0.05, 0.1, 0.2, 0.4],
            "lambda_bridge": [0.0, 0.05, 0.1, 0.2],
            "lambda_path": [0.0],
            "seed_top_s": [20, 30],
            "max_hops": [1, 2],
        }
    )


def graph_rerank_grid_from_record(record: Mapping[str, object]) -> list[GraphRerankConfig]:
    if "type_weights" in record:
        raise ValueError("type_weights is deprecated; use neighbor_type_weights instead.")
    parameters: dict[str, Sequence[object]] = {
        "lambda_init": _candidate_float_values(record, "lambda_init"),
        "lambda_query": _candidate_float_values(record, "lambda_query"),
        "lambda_neighbor": _candidate_float_values(record, "lambda_neighbor"),
        "lambda_bridge": _candidate_float_values(record, "lambda_bridge"),
        "lambda_path": _candidate_float_values(record, "lambda_path"),
        "seed_top_s": _candidate_int_values(record, "seed_top_s"),
        "max_hops": _candidate_int_values(record, "max_hops"),
    }
    neighbor_type_weights = record.get("neighbor_type_weights")
    fixed = (
        {"neighbor_type_weights": neighbor_type_weights}
        if neighbor_type_weights is not None
        else {}
    )
    configs: list[GraphRerankConfig] = []
    for candidate in ParameterGrid(parameters=parameters, fixed=fixed).expand():
        configs.append(ensure_graph_rerank_config(candidate))
    return configs


def _candidate_values(record: Mapping[str, object], key: str) -> Sequence[object]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Graph rerank grid config requires a non-empty list for {key}.")
    return cast(list[object], value)


def _candidate_float_values(record: Mapping[str, object], key: str) -> list[float]:
    return [_finite_float(value, key) for value in _candidate_values(record, key)]


def _candidate_int_values(record: Mapping[str, object], key: str) -> list[int]:
    values: list[int] = []
    for value in _candidate_values(record, key):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Graph rerank grid config values for {key} must be integers.")
        values.append(value)
    return values


def _finite_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Graph rerank grid config values for {key} must be finite numbers.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Graph rerank grid config values for {key} must be finite numbers.")
    return number
