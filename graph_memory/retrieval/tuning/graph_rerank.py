from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import SuiteMetricRow
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.suites import evidence_metric_suite
from graph_memory.registry import Registry
from graph_memory.registry.methods import RetrievalLifecycle
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig, GraphRerankConfigRecord, TuningCandidateRow
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime, TextRankingRequest
from graph_memory.retrieval.tuning.graph_rerank_grid import graph_rerank_grid
from graph_memory.retrieval.tuning.seed_scores import (
    precompute_seed_score_cache,
    run_graph_rerank_from_seed_score_cache,
)
from graph_memory.retrieval.tuning.selection import MetricSelectionKey, retrieval_candidate_key
from graph_memory.tuning.grid_search import GridSearchRunner


class GraphRerankMetricSuite(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, request: EvidenceEvaluationRequest) -> Sequence[SuiteMetricRow]: ...


def tune_graph_rerank(
    *,
    method: str,
    ranking_requests: list[TextRankingRequest],
    labels: list[EvidenceLabel],
    graphs: list[MemoryGraph],
    grid: list[GraphRerankConfig] | None = None,
    top_k: int = 10,
    dense_runtime: DenseRuntime | None = None,
    metric_suite: GraphRerankMetricSuite | None = None,
    selection_key: MetricSelectionKey | None = None,
) -> tuple[GraphRerankConfigRecord, list[TuningCandidateRow]]:
    effective_metric_suite = metric_suite or evidence_metric_suite()
    effective_selection_key = selection_key or retrieval_candidate_key

    graph_rerank_methods = {
        method_id.value
        for method_id in Registry.methods.list_by_lifecycle(RetrievalLifecycle.GRAPH_RERANK)
    }
    if method not in graph_rerank_methods:
        raise ValueError(f"Tuning requires a graph rerank method, got method={method}.")

    definition = Registry.methods.get(method)
    if definition.seed_method is None:
        raise ValueError(f"Graph rerank tuning requires a seed method, got method={method}.")
    seed_score_cache = precompute_seed_score_cache(
        seed_method=RetrievalMethodId(definition.seed_method.value),
        ranking_requests=ranking_requests,
        dense_runtime=dense_runtime or DenseRuntime(config=DenseConfig()),
    )

    def evaluate_candidate(config: GraphRerankConfig) -> TuningCandidateRow:
        config_dict = _graph_rerank_config_record(config)
        predictions = run_graph_rerank_from_seed_score_cache(
            method=method,
            ranking_requests=ranking_requests,
            graphs=graphs,
            seed_score_cache=seed_score_cache,
            top_k=top_k,
            graph_config=config,
        )
        metric_rows = list(
            effective_metric_suite.evaluate(
                EvidenceEvaluationRequest(predictions=predictions, labels=labels, graphs=graphs)
            )
        )
        if len(metric_rows) != 1:
            raise ValueError("Expected one aggregate metric row per tuning candidate.")
        return cast(TuningCandidateRow, cast(object, {**metric_rows[0], "config": config_dict}))

    result = GridSearchRunner[
        GraphRerankConfig,
        TuningCandidateRow,
        tuple[float, ...],
    ](selection_key=effective_selection_key).run(
        grid or graph_rerank_grid(),
        evaluate_candidate,
    )
    candidate_rows = [candidate.evaluation for candidate in result.candidates]
    return result.selected.evaluation["config"], candidate_rows


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
