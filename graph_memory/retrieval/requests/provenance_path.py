from __future__ import annotations

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.graphs.provenance import (
    ARGUMENT_CHUNK_NODE,
    OUTPUT_CHUNK_NODE,
    ProvenanceGraph,
)
from graph_memory.retrieval.requests.text import TextCandidate


class ExecutionProvenanceRankingRequest(DomainModel):
    task_id: NonEmptyStr
    query_text: str
    candidates: tuple[TextCandidate, ...]
    graph: ProvenanceGraph

    @model_validator(mode="after")
    def _validate_context(self) -> "ExecutionProvenanceRankingRequest":
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("execution provenance candidate IDs must be unique")
        graph_candidate_ids = {
            node.node_id
            for node in self.graph.nodes
            if node.kind in {ARGUMENT_CHUNK_NODE, OUTPUT_CHUNK_NODE}
        }
        if set(candidate_ids) != graph_candidate_ids:
            missing = sorted(graph_candidate_ids - set(candidate_ids))
            extra = sorted(set(candidate_ids) - graph_candidate_ids)
            raise ValueError(
                "execution provenance candidates must cover graph content units exactly; "
                f"missing={missing} extra={extra}"
            )
        graph_ids = {
            value
            for candidate in self.candidates
            if isinstance((value := candidate.metadata.get("graph_id")), str)
        }
        if graph_ids and graph_ids != {self.graph.graph_id}:
            raise ValueError(
                "execution provenance candidate graph metadata does not match graph"
            )
        return self


class ProvenancePathRequest(ExecutionProvenanceRankingRequest):
    pass


class ProvenanceRgcnRequest(ExecutionProvenanceRankingRequest):
    pass


__all__ = [
    "ExecutionProvenanceRankingRequest",
    "ProvenancePathRequest",
    "ProvenanceRgcnRequest",
]
