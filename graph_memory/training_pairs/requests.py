from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.requests import TextRankingRequest


@dataclass(frozen=True)
class TrainPairBuildTask:
    text_request: TextRankingRequest
    label: EvidenceLabel
    graph: MemoryGraph


__all__ = ["TrainPairBuildTask"]
