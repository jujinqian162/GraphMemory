from __future__ import annotations

import time
from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.execution.results import assemble_ranked_result
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    MemoryStreamScoringConfigRecord,
    memory_stream_scoring_config_record,
)
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.memory_stream.scoring import (
    NormalizedMemoryStreamSignals,
    RawMemoryStreamSignals,
    normalize_memory_stream_signals,
    normalize_task_signal,
    pseudo_recency_scores,
    rank_memory_stream_scores,
    score_memory_stream,
)
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.retrieval.tuning.seed_scores import precompute_seed_score_cache
from graph_memory.retrieval.tuning.selection import retrieval_candidate_key
from graph_memory.tuning.grid_search import GridSearchRunner
from graph_memory.validation import (
    select_importance_records,
    validate_memory_task_inputs,
    validate_ranked_results,
)


class MemoryStreamTuningCandidateRow(MetricRow):
    config: MemoryStreamScoringConfigRecord


@dataclass(frozen=True)
class MemoryStreamSignalCache:
    relevance_by_task_id: dict[str, dict[str, float]]
    importance_by_task_id: dict[str, dict[str, float]]
    seed_latency_ms_by_task_id: dict[str, float]


def precompute_memory_stream_signal_cache(
    *,
    task_inputs: list[MemoryTaskInput],
    importance_artifact: ImportanceArtifact,
    dense_runtime: DenseRuntime,
) -> MemoryStreamSignalCache:
    seed_cache = precompute_seed_score_cache(
        seed_method=RetrievalMethodId.DENSE,
        task_inputs=task_inputs,
        dense_runtime=dense_runtime,
    )
    importance_records = select_importance_records(importance_artifact, task_inputs)
    importance_by_task_id = {
        record["task_id"]: {
            node_id: float(score)
            for node_id, score in record["scores"].items()
        }
        for record in importance_records
    }

    normalized_relevance: dict[str, dict[str, float]] = {}
    normalized_importance: dict[str, dict[str, float]] = {}
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        normalized = normalize_memory_stream_signals(
            RawMemoryStreamSignals(
                relevance_by_node_id=seed_cache.scores_by_task_id[task_id],
                recency_by_node_id={},
                importance_by_node_id=importance_by_task_id[task_id],
            )
        )
        normalized_relevance[task_id] = dict(
            normalized.relevance_by_node_id
        )
        normalized_importance[task_id] = dict(
            normalized.importance_by_node_id
        )

    return MemoryStreamSignalCache(
        relevance_by_task_id=normalized_relevance,
        importance_by_task_id=normalized_importance,
        seed_latency_ms_by_task_id=dict(seed_cache.latency_ms_by_task_id),
    )


def run_memory_stream_from_signal_cache(
    *,
    task_inputs: list[MemoryTaskInput],
    signal_cache: MemoryStreamSignalCache,
    top_k: int,
    scoring: MemoryStreamScoringConfig,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    validate_memory_task_inputs(task_inputs)
    inputs_by_task_id = {
        task_input["task_id"]: task_input for task_input in task_inputs
    }

    predictions: list[RankedResult] = []
    for task_input in task_inputs:
        task_id = task_input["task_id"]
        try:
            relevance = signal_cache.relevance_by_task_id[task_id]
            importance = signal_cache.importance_by_task_id[task_id]
        except KeyError as error:
            raise ValueError(
                f"Missing Memory Stream signal cache for task_id={task_id}."
            ) from error

        started = time.perf_counter()
        recency = normalize_task_signal(
            pseudo_recency_scores(
                task_input,
                decay=scoring.recency_decay,
            )
        )
        scores = score_memory_stream(
            NormalizedMemoryStreamSignals(
                relevance_by_node_id=relevance,
                recency_by_node_id=recency,
                importance_by_node_id=importance,
            ),
            config=scoring,
        )
        ranked_nodes = [
            RankedNode(node_id=node_id, score=score)
            for node_id, score in rank_memory_stream_scores(scores)
        ]
        ranking_latency_ms = (time.perf_counter() - started) * 1000.0
        predictions.append(
            assemble_ranked_result(
                task_input=task_input,
                method=RetrievalMethodId.MEMORY_STREAM.value,
                ranked_nodes=ranked_nodes,
                top_k=top_k,
                latency_ms=(
                    signal_cache.seed_latency_ms_by_task_id.get(task_id, 0.0)
                    + ranking_latency_ms
                ),
                retrieved_edges=[],
            )
        )

    validate_ranked_results(predictions, inputs_by_task_id)
    return predictions


def tune_memory_stream(
    *,
    task_inputs: list[MemoryTaskInput],
    labels: list[MemoryTaskLabels],
    graphs: list[MemoryGraph],
    importance_artifact: ImportanceArtifact,
    grid: list[MemoryStreamScoringConfig],
    top_k: int = 10,
    dense_runtime: DenseRuntime | None = None,
) -> tuple[
    MemoryStreamScoringConfigRecord,
    list[MemoryStreamTuningCandidateRow],
]:
    from graph_memory.evaluation.service import evaluate_results

    signal_cache = precompute_memory_stream_signal_cache(
        task_inputs=task_inputs,
        importance_artifact=importance_artifact,
        dense_runtime=dense_runtime or DenseRuntime(config=DenseConfig()),
    )

    def evaluate_candidate(
        scoring: MemoryStreamScoringConfig,
    ) -> MemoryStreamTuningCandidateRow:
        predictions = run_memory_stream_from_signal_cache(
            task_inputs=task_inputs,
            signal_cache=signal_cache,
            top_k=top_k,
            scoring=scoring,
        )
        metric_rows = evaluate_results(predictions, labels, graphs)
        if len(metric_rows) != 1:
            raise ValueError(
                "Expected one aggregate metric row per tuning candidate."
            )
        return {
            **metric_rows[0],
            "config": memory_stream_scoring_config_record(scoring),
        }

    result = GridSearchRunner[
        MemoryStreamScoringConfig,
        MemoryStreamTuningCandidateRow,
        tuple[float, float, float, float],
    ](selection_key=retrieval_candidate_key).run(
        grid,
        evaluate_candidate,
    )
    candidate_rows = [
        candidate.evaluation for candidate in result.candidates
    ]
    return result.selected.evaluation["config"], candidate_rows


__all__ = [
    "MemoryStreamSignalCache",
    "MemoryStreamTuningCandidateRow",
    "precompute_memory_stream_signal_cache",
    "run_memory_stream_from_signal_cache",
    "tune_memory_stream",
]
