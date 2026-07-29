from __future__ import annotations

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.requests import TextRankingRequest


class TrainPairBuildTask(DomainModel):
    text_request: TextRankingRequest
    label: EvidenceLabel
    graph: EvidenceGraph | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> "TrainPairBuildTask":
        task_id = self.text_request.task_id
        if self.label.task_id != task_id:
            raise ValueError("train-pair request and label task IDs must match")
        if self.graph is not None and self.graph.task_id != task_id:
            raise ValueError("train-pair request and graph task IDs must match")
        return self


__all__ = ["TrainPairBuildTask"]
