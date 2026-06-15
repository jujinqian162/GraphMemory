from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graph_memory.contracts.common import NodeId
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig


@dataclass(frozen=True)
class RawMemoryStreamSignals:
    relevance_by_node_id: Mapping[NodeId, float]
    recency_by_node_id: Mapping[NodeId, float]
    importance_by_node_id: Mapping[NodeId, float]


@dataclass(frozen=True)
class NormalizedMemoryStreamSignals:
    relevance_by_node_id: Mapping[NodeId, float]
    recency_by_node_id: Mapping[NodeId, float]
    importance_by_node_id: Mapping[NodeId, float]


def normalize_task_signal(raw_scores_by_node_id: Mapping[NodeId, float]) -> dict[NodeId, float]:
    """Map one task-local signal to 0..1, with constant signals mapped to 0.0."""
    if not raw_scores_by_node_id:
        return {}
    min_score = min(raw_scores_by_node_id.values())
    max_score = max(raw_scores_by_node_id.values())
    if max_score == min_score:
        return {node_id: 0.0 for node_id in raw_scores_by_node_id}
    scale = max_score - min_score
    return {
        node_id: (score - min_score) / scale
        for node_id, score in raw_scores_by_node_id.items()
    }


def normalize_memory_stream_signals(raw_signals: RawMemoryStreamSignals) -> NormalizedMemoryStreamSignals:
    """Normalize relevance, pseudo-recency, and importance independently."""
    node_ids = _all_node_ids(raw_signals)
    return NormalizedMemoryStreamSignals(
        relevance_by_node_id=normalize_task_signal(_zero_fill(raw_signals.relevance_by_node_id, node_ids)),
        recency_by_node_id=normalize_task_signal(_zero_fill(raw_signals.recency_by_node_id, node_ids)),
        importance_by_node_id=normalize_task_signal(_zero_fill(raw_signals.importance_by_node_id, node_ids)),
    )


def pseudo_recency_scores(
    task_input: MemoryTaskInput,
    *,
    decay: float,
) -> dict[NodeId, float]:
    """Compute decay ** (max_position - position) for each memory item."""
    memory_items = task_input["memory_items"]
    max_position = max(item["position"] for item in memory_items)
    return {
        memory_item["id"]: decay ** (max_position - memory_item["position"])
        for memory_item in memory_items
    }


def score_memory_stream(
    normalized_signals: NormalizedMemoryStreamSignals,
    *,
    config: MemoryStreamScoringConfig,
) -> dict[NodeId, float]:
    """Return weighted final score per node id."""
    node_ids = _all_node_ids(normalized_signals)
    return {
        node_id: (
            config.relevance_weight
            * normalized_signals.relevance_by_node_id.get(node_id, 0.0)
            + config.recency_weight
            * normalized_signals.recency_by_node_id.get(node_id, 0.0)
            + config.importance_weight
            * normalized_signals.importance_by_node_id.get(node_id, 0.0)
        )
        for node_id in node_ids
    }


def rank_memory_stream_scores(score_by_node_id: Mapping[NodeId, float]) -> list[tuple[NodeId, float]]:
    """Sort by descending score and ascending node id."""
    return sorted(score_by_node_id.items(), key=lambda item: (-item[1], item[0]))


def _all_node_ids(signals: RawMemoryStreamSignals | NormalizedMemoryStreamSignals) -> set[NodeId]:
    return set(signals.relevance_by_node_id) | set(signals.recency_by_node_id) | set(signals.importance_by_node_id)


def _zero_fill(scores_by_node_id: Mapping[NodeId, float], node_ids: set[NodeId]) -> dict[NodeId, float]:
    return {node_id: scores_by_node_id.get(node_id, 0.0) for node_id in node_ids}


__all__ = [
    "NormalizedMemoryStreamSignals",
    "RawMemoryStreamSignals",
    "normalize_memory_stream_signals",
    "normalize_task_signal",
    "pseudo_recency_scores",
    "rank_memory_stream_scores",
    "score_memory_stream",
]
