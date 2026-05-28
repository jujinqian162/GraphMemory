from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from graph_memory.rerank_config import ensure_graph_rerank_config
from graph_memory.retrieval_registry import get_graph_rerank_methods
from graph_memory.retrieval import precompute_initial_score_cache, run_graph_rerank_from_initial_score_cache
from graph_memory.types import (
    GraphRerankConfig,
    GraphRerankConfigRecord,
    MemoryGraph,
    MemoryTaskInput,
    MemoryTaskLabels,
    MetricRow,
    TuningCandidateRow,
)


def tuning_objective(row: MetricRow) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )


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


def select_best_config(rows: list[TuningCandidateRow]) -> GraphRerankConfigRecord:
    if not rows:
        raise ValueError("Cannot select best graph rerank config from empty rows.")

    def sort_key(row: TuningCandidateRow) -> tuple[float, float, float, float]:
        return (
            tuning_objective(row),
            float(row.get("Full Support@10", 0.0)),
            -float(row.get("Retrieval Latency / Query", 0.0)),
            -float(row.get("Avg Retrieved Edges", 0.0)),
        )

    best_row = max(rows, key=sort_key)
    return best_row["config"]


def tune_graph_rerank(
    *,
    method: str,
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    grid: list[GraphRerankConfig] | None = None,
    encoder_model: str = "intfloat/e5-base-v2",
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
    top_k: int = 10,
    dense_encoder: Any | None = None,
) -> tuple[GraphRerankConfigRecord, list[TuningCandidateRow]]:
    from graph_memory.evaluation import evaluate_results

    if method not in get_graph_rerank_methods():
        raise ValueError(f"Tuning requires a graph rerank method, got method={method}.")

    candidate_rows: list[TuningCandidateRow] = []
    initial_score_cache = precompute_initial_score_cache(
        method=method,
        task_inputs=task_inputs,
        encoder_model=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        dense_encoder=dense_encoder,
    )
    for config in grid or graph_rerank_grid():
        config_dict = _graph_rerank_config_record(config)
        predictions = run_graph_rerank_from_initial_score_cache(
            method=method,
            task_inputs=task_inputs,
            graphs=graphs,
            initial_score_cache=initial_score_cache,
            top_k=top_k,
            graph_config=config,
        )
        metric_rows = evaluate_results(predictions, labels, graphs)
        if len(metric_rows) != 1:
            raise ValueError("Expected one aggregate metric row per tuning candidate.")
        candidate_row: TuningCandidateRow = {**metric_rows[0], "config": config_dict}
        candidate_rows.append(candidate_row)
    return select_best_config(candidate_rows), candidate_rows


def _graph_rerank_config_record(config: GraphRerankConfig) -> GraphRerankConfigRecord:
    return {
        "lambda_init": config.lambda_init,
        "lambda_query": config.lambda_query,
        "lambda_neighbor": config.lambda_neighbor,
        "lambda_bridge": config.lambda_bridge,
        "lambda_path": config.lambda_path,
        "seed_top_s": config.seed_top_s,
        "max_hops": config.max_hops,
        "neighbor_type_weights": dict(config.neighbor_type_weights),
    }
