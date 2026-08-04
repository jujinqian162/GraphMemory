from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from pydantic import Field, model_validator

from graph_memory.contracts.model import DomainModel
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.training_pairs.contracts import TrainPairDataset, TrainPairRecord

if TYPE_CHECKING:
    from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
    from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TrainDependencies:
    text_embedding_provider: "TextEmbeddingProvider"
    seed_signal_provider: "SeedSignalProvider"


class RgcnTrainPayload(DomainModel):
    train_requests: tuple[TextRankingRequest, ...]
    train_labels: tuple[EvidenceLabel, ...]
    train_graphs: tuple[EvidenceGraph, ...]
    train_pairs: tuple[TrainPairRecord, ...]
    dev_requests: tuple[TextRankingRequest, ...]
    dev_labels: tuple[EvidenceLabel, ...]
    dev_graphs: tuple[EvidenceGraph, ...]

    @model_validator(mode="after")
    def _validate_training_input(self) -> "RgcnTrainPayload":
        TrainPairDataset(
            requests=self.train_requests,
            labels=self.train_labels,
            graphs=self.train_graphs,
            pairs=self.train_pairs,
        )
        _validate_dev(self.dev_requests, self.dev_labels, self.dev_graphs)
        return self


class DenseFinetuneTrainPayload(DomainModel):
    train_requests: tuple[TextRankingRequest, ...]
    train_labels: tuple[EvidenceLabel, ...]
    train_pairs: tuple[TrainPairRecord, ...]
    train_group_ids: dict[str, str] = Field(default_factory=dict)
    dev_requests: tuple[TextRankingRequest, ...]
    dev_labels: tuple[EvidenceLabel, ...]
    dev_query_origins: dict[str, str] = Field(default_factory=dict)
    output_dir: Path
    model_dir: Path

    @model_validator(mode="after")
    def _validate_training_input(self) -> "DenseFinetuneTrainPayload":
        TrainPairDataset(
            requests=self.train_requests,
            labels=self.train_labels,
            pairs=self.train_pairs,
        )
        _validate_dev(self.dev_requests, self.dev_labels, ())
        _validate_group_ids(
            self.train_requests,
            self.train_group_ids,
            name="train",
        )
        _validate_group_ids(
            self.dev_requests,
            self.dev_query_origins,
            name="dev query origins",
        )
        return self


TrainPayload: TypeAlias = RgcnTrainPayload | DenseFinetuneTrainPayload


def _validate_group_ids(
    requests: tuple[TextRankingRequest, ...],
    group_ids: dict[str, str],
    *,
    name: str,
) -> None:
    if group_ids and set(group_ids) != {request.task_id for request in requests}:
        raise ValueError(f"dense-ft {name} group IDs must align with requests")


def _validate_dev(
    requests: tuple[TextRankingRequest, ...],
    labels: tuple[EvidenceLabel, ...],
    graphs: tuple[EvidenceGraph, ...],
) -> None:
    request_ids = [request.task_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("dev request task IDs must be unique")
    label_ids = [label.task_id for label in labels]
    if set(request_ids) != set(label_ids) or len(label_ids) != len(set(label_ids)):
        raise ValueError("dev requests and labels must align")
    if graphs:
        graph_ids = [graph.task_id for graph in graphs]
        if set(request_ids) != set(graph_ids) or len(graph_ids) != len(set(graph_ids)):
            raise ValueError("dev requests and graphs must align")


__all__ = [
    "DenseFinetuneTrainPayload",
    "RgcnTrainPayload",
    "TrainDependencies",
    "TrainPayload",
]
