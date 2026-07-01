from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import EvidenceMetricRow, LongMemEvalMetricRow, SuiteMetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.suites import evidence_metric_suite
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
from graph_memory.retrieval.methods.memory_stream.importance import request_importance_scores
from graph_memory.retrieval.methods.memory_stream.scoring import (
    NormalizedMemoryStreamSignals,
    RawMemoryStreamSignals,
    normalize_memory_stream_signals,
    normalize_task_signal,
    memory_stream_recency_scores,
    rank_memory_stream_scores,
    score_memory_stream,
)
from graph_memory.retrieval.requests import (
    DenseRuntime,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)
from graph_memory.retrieval.tuning.seed_scores import precompute_seed_score_cache
from graph_memory.retrieval.tuning.selection import MetricSelectionKey, retrieval_candidate_key
from graph_memory.tuning.grid_search import GridSearchRunner
from graph_memory.validation import (
    select_importance_records,
    validate_ranked_results,
)


class EvidenceMemoryStreamTuningCandidateRow(EvidenceMetricRow):
    config: MemoryStreamScoringConfigRecord


class LongMemEvalMemoryStreamTuningCandidateRow(LongMemEvalMetricRow):
    config: MemoryStreamScoringConfigRecord


MemoryStreamTuningCandidateRow: TypeAlias = (
    EvidenceMemoryStreamTuningCandidateRow | LongMemEvalMemoryStreamTuningCandidateRow
)


class MemoryStreamMetricSuite(Protocol):
    def evaluate(self, request: EvidenceEvaluationRequest) -> Sequence[SuiteMetricRow]: ...


@dataclass(frozen=True)
class MemoryStreamSignalCache:
    relevance_by_task_id: dict[str, dict[str, float]]
    importance_by_task_id: dict[str, dict[str, float]]
    seed_latency_ms_by_task_id: dict[str, float]


def precompute_memory_stream_signal_cache(
    *,
    temporal_requests: list[TemporalMemoryRankingRequest],
    importance_artifact: ImportanceArtifact | None,
    dense_runtime: DenseRuntime,
    require_complete_request_importance: bool = False,
) -> MemoryStreamSignalCache:
    text_requests = [_text_request_from_temporal(request) for request in temporal_requests]
    seed_cache = precompute_seed_score_cache(
        seed_method=RetrievalMethodId.DENSE,
        ranking_requests=text_requests,
        dense_runtime=dense_runtime,
    )
    importance_by_task_id = _importance_by_task_id(
        temporal_requests=temporal_requests,
        importance_artifact=importance_artifact,
        require_complete_request_importance=require_complete_request_importance,
    )

    normalized_relevance: dict[str, dict[str, float]] = {}
    normalized_importance: dict[str, dict[str, float]] = {}
    for request in temporal_requests:
        task_id = request.task_id
        normalized = normalize_memory_stream_signals(
            RawMemoryStreamSignals(
                relevance_by_node_id=seed_cache.scores_by_task_id[task_id],
                recency_by_node_id={},
                importance_by_node_id=importance_by_task_id[task_id],
            )
        )
        normalized_relevance[task_id] = dict(normalized.relevance_by_node_id)
        normalized_importance[task_id] = dict(normalized.importance_by_node_id)

    return MemoryStreamSignalCache(
        relevance_by_task_id=normalized_relevance,
        importance_by_task_id=normalized_importance,
        seed_latency_ms_by_task_id=dict(seed_cache.latency_ms_by_task_id),
    )


def run_memory_stream_from_signal_cache(
    *,
    temporal_requests: list[TemporalMemoryRankingRequest],
    signal_cache: MemoryStreamSignalCache,
    top_k: int,
    scoring: MemoryStreamScoringConfig,
) -> list[RankedResult]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    predictions: list[RankedResult] = []
    for request in temporal_requests:
        task_id = request.task_id
        try:
            relevance = signal_cache.relevance_by_task_id[task_id]
            importance = signal_cache.importance_by_task_id[task_id]
        except KeyError as error:
            raise ValueError(
                f"Missing Memory Stream signal cache for task_id={task_id}."
            ) from error

        started = time.perf_counter()
        recency = (
            normalize_task_signal(
                memory_stream_recency_scores(
                    request,
                    decay=scoring.recency_decay,
                )
            )
            if scoring.recency_weight > 0.0
            else {}
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
                text_request=_text_request_from_temporal(request),
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

    validate_ranked_results(predictions, [_text_request_from_temporal(request) for request in temporal_requests])
    return predictions


def tune_memory_stream(
    *,
    temporal_requests: list[TemporalMemoryRankingRequest],
    labels: list[EvidenceLabel],
    graphs: list[MemoryGraph],
    importance_artifact: ImportanceArtifact | None,
    grid: list[MemoryStreamScoringConfig],
    top_k: int = 10,
    dense_runtime: DenseRuntime | None = None,
    metric_suite: MemoryStreamMetricSuite | None = None,
    selection_key: MetricSelectionKey | None = None,
) -> tuple[
    MemoryStreamScoringConfigRecord,
    list[MemoryStreamTuningCandidateRow],
]:
    effective_metric_suite = metric_suite or evidence_metric_suite()
    effective_selection_key = selection_key or retrieval_candidate_key

    signal_cache = precompute_memory_stream_signal_cache(
        temporal_requests=temporal_requests,
        importance_artifact=importance_artifact,
        dense_runtime=dense_runtime or DenseRuntime(config=DenseConfig()),
        require_complete_request_importance=(
            importance_artifact is None
            and any(config.importance_weight > 0.0 for config in grid)
        ),
    )

    def evaluate_candidate(
        scoring: MemoryStreamScoringConfig,
    ) -> MemoryStreamTuningCandidateRow:
        predictions = run_memory_stream_from_signal_cache(
            temporal_requests=temporal_requests,
            signal_cache=signal_cache,
            top_k=top_k,
            scoring=scoring,
        )
        metric_rows = list(
            effective_metric_suite.evaluate(
                EvidenceEvaluationRequest(predictions=predictions, labels=labels, graphs=graphs)
            )
        )
        if len(metric_rows) != 1:
            raise ValueError(
                "Expected one aggregate metric row per tuning candidate."
            )
        return cast(
            MemoryStreamTuningCandidateRow,
            cast(
                object,
                {
                    **metric_rows[0],
                    "config": memory_stream_scoring_config_record(scoring),
                },
            ),
        )

    result = GridSearchRunner[
        MemoryStreamScoringConfig,
        MemoryStreamTuningCandidateRow,
        tuple[float, ...],
    ](selection_key=effective_selection_key).run(
        grid,
        evaluate_candidate,
    )
    candidate_rows = [candidate.evaluation for candidate in result.candidates]
    return result.selected.evaluation["config"], candidate_rows


def _importance_by_task_id(
    *,
    temporal_requests: list[TemporalMemoryRankingRequest],
    importance_artifact: ImportanceArtifact | None,
    require_complete_request_importance: bool,
) -> dict[str, dict[str, float]]:
    if importance_artifact is not None:
        importance_records = select_importance_records(importance_artifact, temporal_requests)
        return {
            record["task_id"]: {
                node_id: float(score)
                for node_id, score in record["scores"].items()
            }
            for record in importance_records
        }
    return {
        request.task_id: request_importance_scores(
            request,
            require_complete=require_complete_request_importance,
        )
        for request in temporal_requests
    }



def _text_request_from_temporal(request: TemporalMemoryRankingRequest) -> TextRankingRequest:
    return TextRankingRequest(task_id=request.task_id, query_text=request.query_text, candidates=request.candidates)


__all__ = [
    "MemoryStreamMetricSuite",
    "MemoryStreamSignalCache",
    "MemoryStreamTuningCandidateRow",
    "precompute_memory_stream_signal_cache",
    "run_memory_stream_from_signal_cache",
    "tune_memory_stream",
]
