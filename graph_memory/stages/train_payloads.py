from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.requests import TextRankingRequest

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TrainDependencies:
    text_embedding_provider: "TextEmbeddingProvider"
    seed_signal_provider: "SeedSignalProvider"


@dataclass(frozen=True)
class RgcnTrainPayload:
    train_requests: list[TextRankingRequest]
    train_graphs: list[MemoryGraph]
    train_pairs: list[TrainPairRecord]
    dev_requests: list[TextRankingRequest]
    dev_labels: list[EvidenceLabel]
    dev_graphs: list[MemoryGraph]
    train_labels: list[EvidenceLabel] | None = None
    dependencies: TrainDependencies | None = None


@dataclass(frozen=True)
class DenseFinetuneTrainPayload:
    train_requests: list[TextRankingRequest]
    train_labels: list[EvidenceLabel]
    train_pairs: list[TrainPairRecord]
    dev_requests: list[TextRankingRequest]
    dev_labels: list[EvidenceLabel]
    output_dir: Path
    model_dir: Path


TrainPayload: TypeAlias = RgcnTrainPayload | DenseFinetuneTrainPayload


__all__ = [
    "DenseFinetuneTrainPayload",
    "RgcnTrainPayload",
    "TrainDependencies",
    "TrainPayload",
]
