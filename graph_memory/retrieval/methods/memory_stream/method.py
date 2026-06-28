from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, SeedRanker
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.scoring import (
    RawMemoryStreamSignals,
    normalize_memory_stream_signals,
    memory_stream_recency_scores,
    rank_memory_stream_scores,
    score_memory_stream,
)
from graph_memory.retrieval.requests import RankingMethodRequest, TemporalMemoryRankingRequest, TextRankingRequest


@dataclass(frozen=True)
class MemoryStreamMethod:
    name: str
    dense_seed_ranker: SeedRanker
    importance_by_task_id: Mapping[TaskId, TaskImportanceRecord]
    scoring: MemoryStreamScoringConfig

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        _ = top_k
        if not isinstance(request, TemporalMemoryRankingRequest):
            raise TypeError(f"{self.name} requires TemporalMemoryRankingRequest, got {type(request).__name__}.")
        raw_signals = RawMemoryStreamSignals(
            relevance_by_node_id=self._dense_relevance_by_node_id(request),
            recency_by_node_id=(
                memory_stream_recency_scores(request, decay=self.scoring.recency_decay)
                if self.scoring.recency_weight > 0.0
                else {}
            ),
            importance_by_node_id=request.importance_by_item_id,
        )
        normalized_signals = normalize_memory_stream_signals(raw_signals)
        score_by_node_id = score_memory_stream(normalized_signals, config=self.scoring)
        ranked_node_scores = rank_memory_stream_scores(score_by_node_id)
        ranked_nodes = [
            RankedNode(node_id=node_id, score=score)
            for node_id, score in ranked_node_scores
        ]
        return RetrievalMethodResult(ranked_nodes=ranked_nodes)

    def importance_scores_for_task(self, task_id: TaskId) -> dict[NodeId, float]:
        try:
            task_record = self.importance_by_task_id[task_id]
        except KeyError as error:
            raise ValueError(f"Missing importance record for task_id={task_id}.") from error
        return {node_id: float(score) for node_id, score in task_record["scores"].items()}

    def _dense_relevance_by_node_id(self, request: TemporalMemoryRankingRequest) -> dict[NodeId, float]:
        """Use the existing dense seed ranking contract as the relevance source."""
        text_request = TextRankingRequest(
            task_id=request.task_id,
            query_text=request.query_text,
            candidates=request.candidates,
        )
        return {
            ranked_node.node_id: ranked_node.score
            for ranked_node in self.dense_seed_ranker.rank(text_request)
        }


__all__ = ["MemoryStreamMethod"]
