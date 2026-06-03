from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product

from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, ensure_graph_rerank_config


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
    lambda_init_values = _candidate_float_values(record, "lambda_init")
    lambda_query_values = _candidate_float_values(record, "lambda_query")
    lambda_neighbor_values = _candidate_float_values(record, "lambda_neighbor")
    lambda_bridge_values = _candidate_float_values(record, "lambda_bridge")
    lambda_path_values = _candidate_float_values(record, "lambda_path")
    seed_top_s_values = _candidate_int_values(record, "seed_top_s")
    max_hops_values = _candidate_int_values(record, "max_hops")
    neighbor_type_weights = record.get("neighbor_type_weights")
    configs: list[GraphRerankConfig] = []
    for lambda_init, lambda_query, lambda_neighbor, lambda_bridge, lambda_path, seed_top_s, max_hops in product(
        lambda_init_values,
        lambda_query_values,
        lambda_neighbor_values,
        lambda_bridge_values,
        lambda_path_values,
        seed_top_s_values,
        max_hops_values,
    ):
        kwargs: dict[str, object] = {
            "lambda_init": lambda_init,
            "lambda_query": lambda_query,
            "lambda_neighbor": lambda_neighbor,
            "lambda_bridge": lambda_bridge,
            "lambda_path": lambda_path,
            "seed_top_s": seed_top_s,
            "max_hops": max_hops,
        }
        if neighbor_type_weights is not None:
            kwargs["neighbor_type_weights"] = neighbor_type_weights
        configs.append(ensure_graph_rerank_config(kwargs))
    return configs


def _candidate_values(record: Mapping[str, object], key: str) -> Sequence[object]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Graph rerank grid config requires a non-empty list for {key}.")
    return value


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
