from __future__ import annotations

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel, NonEmptyStr
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.requests import TextRankingRequest


class CandidateNeighborEdge(DomainModel):
    source: NonEmptyStr
    target: NonEmptyStr

    @model_validator(mode="after")
    def _reject_self_edge(self) -> "CandidateNeighborEdge":
        if self.source == self.target:
            raise ValueError("candidate neighbor edge cannot be a self edge")
        return self


class TrainPairBuildTask(DomainModel):
    text_request: TextRankingRequest
    label: EvidenceLabel
    graph: EvidenceGraph | None = None
    candidate_neighbor_edges: tuple[CandidateNeighborEdge, ...] | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> "TrainPairBuildTask":
        task_id = self.text_request.task_id
        if self.label.task_id != task_id:
            raise ValueError("train-pair request and label task IDs must match")
        if self.graph is not None and self.graph.task_id != task_id:
            raise ValueError("train-pair request and graph task IDs must match")
        if self.candidate_neighbor_edges is not None:
            keys = [
                (edge.source, edge.target) for edge in self.candidate_neighbor_edges
            ]
            if len(keys) != len(set(keys)):
                raise ValueError("candidate neighbor edges must be unique")
        return self


__all__ = ["CandidateNeighborEdge", "TrainPairBuildTask"]
