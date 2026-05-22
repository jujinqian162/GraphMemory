from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import product
from typing import Any, cast

from graph_memory.retrieval import precompute_initial_score_cache, run_graph_rerank_from_initial_score_cache
from graph_memory.types import (
    GraphRerankConfig,
    GraphRerankConfigRecord,
    MemoryGraph,
    MemoryTaskInput,
    MemoryTaskLabels,
    MetricRow,
    TuningCandidateRow,
    graph_rerank_config_from_value,
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
    values = {
        "lambda_init": _candidate_values(record, "lambda_init"),
        "lambda_query": _candidate_values(record, "lambda_query"),
        "lambda_neighbor": _candidate_values(record, "lambda_neighbor"),
        "lambda_bridge": _candidate_values(record, "lambda_bridge"),
        "lambda_path": _candidate_values(record, "lambda_path"),
        "seed_top_s": _candidate_values(record, "seed_top_s"),
        "max_hops": _candidate_values(record, "max_hops"),
    }
    neighbor_type_weights = record.get("neighbor_type_weights")
    deprecated_type_weights = record.get("type_weights")
    configs: list[GraphRerankConfig] = []
    for lambda_init, lambda_query, lambda_neighbor, lambda_bridge, lambda_path, seed_top_s, max_hops in product(
        values["lambda_init"],
        values["lambda_query"],
        values["lambda_neighbor"],
        values["lambda_bridge"],
        values["lambda_path"],
        values["seed_top_s"],
        values["max_hops"],
    ):
        kwargs: dict[str, Any] = {
            "lambda_init": float(lambda_init),
            "lambda_query": float(lambda_query),
            "lambda_neighbor": float(lambda_neighbor),
            "lambda_bridge": float(lambda_bridge),
            "lambda_path": float(lambda_path),
            "seed_top_s": int(seed_top_s),
            "max_hops": int(max_hops),
        }
        if isinstance(neighbor_type_weights, dict):
            kwargs["neighbor_type_weights"] = {
                str(key): float(value) for key, value in neighbor_type_weights.items()
            }
        elif isinstance(deprecated_type_weights, dict):
            kwargs["type_weights"] = {
                str(key): float(value) for key, value in deprecated_type_weights.items()
            }
        configs.append(graph_rerank_config_from_value(kwargs))
    return configs


def _candidate_values(record: Mapping[str, object], key: str) -> Sequence[object]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Graph rerank grid config requires a non-empty list for {key}.")
    return value


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

    if method not in {"bm25_graph_rerank", "dense_graph_rerank"}:
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
        config_dict = cast(GraphRerankConfigRecord, asdict(config))
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
