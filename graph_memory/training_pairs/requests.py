from __future__ import annotations

from dataclasses import dataclass

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.provenance import ExecutionProvenanceGraph
from graph_memory.retrieval.requests import TextRankingRequest


@dataclass(frozen=True)
class TrainPairBuildTask:
    text_request: TextRankingRequest
    label: EvidenceLabel
    graph: EvidenceGraph | None = None


@dataclass(frozen=True)
class ProvenanceTrainPairBuildTask:
    text_request: TextRankingRequest
    graph: ExecutionProvenanceGraph
    label: EvidenceLabel


__all__ = ["ProvenanceTrainPairBuildTask", "TrainPairBuildTask"]
