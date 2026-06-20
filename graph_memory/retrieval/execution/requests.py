from __future__ import annotations

from dataclasses import dataclass

from graph_memory.retrieval.requests import RankingMethodRequest, TextRankingRequest


@dataclass(frozen=True)
class RetrievalExecutionTask:
    text_request: TextRankingRequest
    method_request: RankingMethodRequest

    def __post_init__(self) -> None:
        if self.text_request.task_id != self.method_request.task_id:
            raise ValueError(
                "Retrieval execution task request mismatch: "
                f"text task_id={self.text_request.task_id}, "
                f"method task_id={self.method_request.task_id}."
            )


__all__ = ["RetrievalExecutionTask"]