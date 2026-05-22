from __future__ import annotations

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
)


def tuning_objective(row: MetricRow) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )


def graph_rerank_grid() -> list[GraphRerankConfig]:
    configs: list[GraphRerankConfig] = []
    for lambda_query, lambda_neighbor, lambda_bridge, seed_top_s, max_hops in product(
        [0.0, 0.05, 0.1, 0.2],
        [0.0, 0.05, 0.1, 0.2, 0.4],
        [0.0, 0.05, 0.1, 0.2],
        [20, 30],
        [1, 2],
    ):
        configs.append(
            GraphRerankConfig(
                lambda_init=1.0,
                lambda_query=lambda_query,
                lambda_neighbor=lambda_neighbor,
                lambda_bridge=lambda_bridge,
                lambda_path=0.0,
                seed_top_s=seed_top_s,
                max_hops=max_hops,
            )
        )
    return configs


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
