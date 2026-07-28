from __future__ import annotations

from pydantic import model_validator

from graph_memory.contracts.model import DomainModel
from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest


class RetrievalExecutionTask(DomainModel):
    text_request: TextRankingRequest
    method_request: RankingMethodRequest

    @model_validator(mode="after")
    def _validate_task_id(self) -> "RetrievalExecutionTask":
        if self.text_request.task_id != self.method_request.task_id:
            raise ValueError(
                "retrieval execution task request mismatch: "
                f"text task_id={self.text_request.task_id}, "
                f"method task_id={self.method_request.task_id}"
            )
        return self


__all__ = ["RetrievalExecutionTask"]
