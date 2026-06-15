from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult, SeedRanker
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.scoring import (
    RawMemoryStreamSignals,
    normalize_memory_stream_signals,
    pseudo_recency_scores,
    rank_memory_stream_scores,
    score_memory_stream,
)


@dataclass(frozen=True)
class MemoryStreamMethod:
    name: str
    dense_seed_ranker: SeedRanker
    importance_by_task_id: Mapping[TaskId, TaskImportanceRecord]
    scoring: MemoryStreamScoringConfig

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> RetrievalMethodResult:
        _ = top_k
        raw_signals = RawMemoryStreamSignals(
            relevance_by_node_id=self._dense_relevance_by_node_id(task_input),
            recency_by_node_id=(
                pseudo_recency_scores(task_input, decay=self.scoring.recency_decay)
                if self.scoring.recency_weight > 0.0
                else {}
            ),
            importance_by_node_id=self._importance_by_node_id(task_input),
        )
        normalized_signals = normalize_memory_stream_signals(raw_signals)
        score_by_node_id = score_memory_stream(normalized_signals, config=self.scoring)
        ranked_node_scores = rank_memory_stream_scores(score_by_node_id)
        ranked_nodes = [
            RankedNode(node_id=node_id, score=score)
            for node_id, score in ranked_node_scores
        ]
        return RetrievalMethodResult(ranked_nodes=ranked_nodes)

    def _dense_relevance_by_node_id(self, task_input: MemoryTaskInput) -> dict[NodeId, float]:
        """Use the existing dense seed ranking contract as the relevance source."""
        return {
            ranked_node.node_id: ranked_node.score
            for ranked_node in self.dense_seed_ranker.rank(task_input)
        }

    def _importance_by_node_id(self, task_input: MemoryTaskInput) -> dict[NodeId, float]:
        """Read cleaned integer importance scores from the prevalidated task record."""
        task_id = task_input["task_id"]
        try:
            task_record = self.importance_by_task_id[task_id]
        except KeyError as error:
            raise ValueError(f"Missing importance record for task_id={task_id}.") from error
        return {node_id: float(score) for node_id, score in task_record["scores"].items()}


__all__ = ["MemoryStreamMethod"]
