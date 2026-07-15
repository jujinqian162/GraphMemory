from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import TaskId
from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.retrieval.requests.text import TextCandidate


@dataclass(frozen=True)
class EvidenceGraphRankingRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    graph: EvidenceGraph
    initial_scores: Mapping[str, float]


__all__ = ["EvidenceGraphRankingRequest"]
