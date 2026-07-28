from __future__ import annotations

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel, FiniteFloat, NonEmptyStr
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.requests.text import TextCandidate


class EvidenceGraphRankingRequest(DomainModel):
    task_id: NonEmptyStr
    query_text: str
    candidates: tuple[TextCandidate, ...]
    graph: EvidenceGraph
    initial_scores: dict[NonEmptyStr, FiniteFloat]

    @model_validator(mode="after")
    def _validate_context(self) -> "EvidenceGraphRankingRequest":
        if self.graph.task_id != self.task_id:
            raise ValueError("evidence graph request task_id must match graph task_id")
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("evidence graph candidate IDs must be unique")
        expected = set(candidate_ids)
        if self.graph.graph_item_ids != expected:
            raise ValueError("evidence graph items must cover candidates exactly")
        if set(self.initial_scores) != expected:
            raise ValueError("initial scores must cover candidates exactly")
        return self


__all__ = ["EvidenceGraphRankingRequest"]
