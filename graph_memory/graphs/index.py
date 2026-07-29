from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import TaskId
from graph_memory.graphs.contracts import EvidenceGraph


@dataclass(frozen=True)
class GraphIndex:
    graph_by_task_id: dict[TaskId, EvidenceGraph]

    @classmethod
    def from_graphs(cls, graphs: Sequence[EvidenceGraph | object]) -> GraphIndex:
        validated = [
            graph
            if isinstance(graph, EvidenceGraph)
            else EvidenceGraph.model_validate(graph)
            for graph in graphs
        ]
        graph_by_task_id = {graph.task_id: graph for graph in validated}
        if len(graph_by_task_id) != len(validated):
            raise ValueError("Evidence graph task IDs must be unique.")
        return cls(graph_by_task_id=graph_by_task_id)

    def get_required(self, task_id: TaskId) -> EvidenceGraph:
        graph = self.graph_by_task_id.get(task_id)
        if graph is None:
            raise ValueError(f"Missing graph for task_id={task_id}.")
        return graph
