from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from abstraction.domain.common.identifiers import ArtifactId, MethodId
from abstraction.domain.graphs.artifacts import GraphArtifact
from abstraction.domain.task_views.views import TrainingView


@dataclass(frozen=True)
class TrainArtifactContract:
    artifact_id: ArtifactId
    method_id: MethodId
    artifact_kind: str
    compatible_request_kinds: Sequence[str]


class TrainingViewBuilder(Protocol):
    def build_training_view(self, source_view: object) -> TrainingView:
        ...


class TrainableMethodAdapter(Protocol):
    def train_method(
        self,
        training_views: Sequence[TrainingView],
        optional_graphs: Sequence[GraphArtifact],
    ) -> TrainArtifactContract:
        ...


class PairBasedTrainingViewBuilder:  # implement TrainingViewBuilder
    def build_training_view(self, source_view: object) -> TrainingView:
        raise NotImplementedError
class DenseFineTuneTrainingAdapter:  # implement TrainableMethodAdapter
    def train_method(
        self,
        training_views: Sequence[TrainingView],
        optional_graphs: Sequence[GraphArtifact],
    ) -> TrainArtifactContract:
        raise NotImplementedError
class GraphRetrieverTrainingAdapter:  # implement TrainableMethodAdapter
    def train_method(
        self,
        training_views: Sequence[TrainingView],
        optional_graphs: Sequence[GraphArtifact],
    ) -> TrainArtifactContract:
        raise NotImplementedError