from __future__ import annotations

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.provenance import ExecutionProvenanceGraph
from graph_memory.retrieval.requests.text import TextCandidate


class ExecutionProvenanceRankingRequest(DomainModel):
    task_id: NonEmptyStr
    query_text: str
    candidates: tuple[TextCandidate, ...]
    graph: ExecutionProvenanceGraph

    @model_validator(mode="after")
    def _validate_context(self) -> "ExecutionProvenanceRankingRequest":
        if self.graph.task_id != self.task_id:
            raise ValueError(
                "execution provenance request task_id must match graph task_id"
            )
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("execution provenance candidate IDs must be unique")
        node_ids = {node.node_id for node in self.graph.nodes}
        missing = sorted(set(candidate_ids) - node_ids)
        if missing:
            raise ValueError(
                f"execution provenance candidates missing from graph: {missing}"
            )
        return self


__all__ = ["ExecutionProvenanceRankingRequest"]
