from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.registry import Registry
from graph_memory.registry.methods import RetrievalLifecycle
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, GraphRerankConfigRecord, TuningCandidateRow
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.retrieval.tuning.initial_scores import (
    precompute_initial_score_cache,
    run_graph_rerank_from_initial_score_cache,
)
from graph_memory.retrieval.tuning.grid import graph_rerank_grid


def tuning_objective(row: MetricRow) -> float:
    return (
        0.50 * float(row["Full Support@5"])
        + 0.30 * float(row["Recall@5"])
        + 0.20 * float(row["Connected Evidence Recall@10"])
    )


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
    top_k: int = 10,
    dense_runtime: DenseRuntime | None = None,
) -> tuple[GraphRerankConfigRecord, list[TuningCandidateRow]]:
    from graph_memory.evaluation.service import evaluate_results

    graph_rerank_methods = {
        method_id.value
        for method_id in Registry.methods.list_by_lifecycle(RetrievalLifecycle.GRAPH_RERANK)
    }
    if method not in graph_rerank_methods:
        raise ValueError(f"Tuning requires a graph rerank method, got method={method}.")

    candidate_rows: list[TuningCandidateRow] = []
    initial_score_cache = precompute_initial_score_cache(
        method=method,
        task_inputs=task_inputs,
        dense_runtime=dense_runtime or DenseRuntime(config=DenseConfig()),
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
