from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import TaskId
from graph_memory.graphs.provenance import ExecutionProvenanceGraph
from graph_memory.retrieval.requests.text import TextCandidate


@dataclass(frozen=True)
class ExecutionProvenanceRankingRequest:
    task_id: TaskId
    query_text: str
    candidates: Sequence[TextCandidate]
    graph: ExecutionProvenanceGraph

    def __post_init__(self) -> None:
        if self.graph.task_id != self.task_id:
            raise ValueError(
                "Execution provenance request task_id must match graph task_id."
            )
        node_ids = {node.node_id for node in self.graph.nodes}
        missing = sorted(
            candidate.item_id
            for candidate in self.candidates
            if candidate.item_id not in node_ids
        )
        if missing:
            raise ValueError(
                f"Execution provenance candidates missing from graph: {missing}"
            )


__all__ = ["ExecutionProvenanceRankingRequest"]
