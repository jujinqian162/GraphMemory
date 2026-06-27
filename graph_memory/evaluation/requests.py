from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.contracts.common import NodeId, TaskId
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult


@dataclass(frozen=True)
class EvidenceLabel:
    task_id: TaskId
    gold_answer: str
    gold_evidence_item_ids: tuple[NodeId, ...]
    gold_dependency_edges: tuple[tuple[NodeId, NodeId], ...]
    gold_session_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceEvaluationRequest:
    predictions: Sequence[RankedResult]
    labels: Sequence[EvidenceLabel]
    graphs: Sequence[MemoryGraph]


__all__ = ["EvidenceEvaluationRequest", "EvidenceLabel"]