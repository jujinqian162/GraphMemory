from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeAlias, TypeVar

from abstraction.domain.common.capability_names import PredictionKind, RequestKind, ViewKind
from abstraction.domain.common.identifiers import DatasetId
from abstraction.domain.graphs.artifacts import GraphArtifact
from abstraction.domain.projections.eval_units import EvaluationUnitBatch
from abstraction.domain.retrieval.predictions import RetrievalPrediction
from abstraction.domain.retrieval.requests import RetrievalRequest
from abstraction.domain.task_views.eval_views import EvalLabelView
from abstraction.domain.task_views.views import TaskView

SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")


@dataclass(frozen=True)
class ProjectionDefinition:
    source_context: str
    target_context: str
    source_kind: ViewKind | PredictionKind
    target_kind: RequestKind | str
    dataset_scope: DatasetId | None


class ProjectionAdapter(Protocol, Generic[SourceT, TargetT]):
    def describe_projection(self) -> ProjectionDefinition:
        ...

    def project(self, source: SourceT) -> TargetT:
        ...


PredictionEvalProjectionSource: TypeAlias = tuple[RetrievalPrediction, EvalLabelView, GraphArtifact | None]
TaskToRequestProjection: TypeAlias = ProjectionAdapter[TaskView, RetrievalRequest]
PredictionToEvalProjection: TypeAlias = ProjectionAdapter[PredictionEvalProjectionSource, EvaluationUnitBatch]
RegisteredProjection: TypeAlias = TaskToRequestProjection | PredictionToEvalProjection


class ProjectionRegistry(Protocol):
    def register_projection(self, adapter: RegisteredProjection) -> None:
        ...

    def find_task_to_request_projection(
        self,
        source_view_kind: ViewKind,
        target_request_kind: RequestKind,
        dataset_id: DatasetId,
    ) -> TaskToRequestProjection:
        ...

    def find_prediction_to_eval_projection(
        self,
        source_prediction_kind: PredictionKind,
        target_eval_unit_kind: str,
        dataset_id: DatasetId,
    ) -> PredictionToEvalProjection:
        ...


class ProjectionPlanner(Protocol):
    def plan_projection_chain(self, requested_projection: ProjectionDefinition) -> Sequence[ProjectionDefinition]:
        ...


class CapabilityProjectionRegistry:  # implement ProjectionRegistry
    def __init__(self) -> None:
        self.projections: list[RegisteredProjection] = []

    def register_projection(self, adapter: RegisteredProjection) -> None:
        self.projections.append(adapter)

    def find_task_to_request_projection(
        self,
        source_view_kind: ViewKind,
        target_request_kind: RequestKind,
        dataset_id: DatasetId,
    ) -> TaskToRequestProjection:
        for projection in self.projections:
            definition = projection.describe_projection()
            if definition.source_kind == source_view_kind and definition.target_kind == target_request_kind:
                if definition.dataset_scope in (None, dataset_id):
                    return projection
        pass

    def find_prediction_to_eval_projection(
        self,
        source_prediction_kind: PredictionKind,
        target_eval_unit_kind: str,
        dataset_id: DatasetId,
    ) -> PredictionToEvalProjection:
        for projection in self.projections:
            definition = projection.describe_projection()
            if definition.source_kind == source_prediction_kind and definition.target_kind == target_eval_unit_kind:
                if definition.dataset_scope in (None, dataset_id):
                    return projection
        pass


class RegistryProjectionPlanner:  # implement ProjectionPlanner
    def __init__(self, projection_registry: ProjectionRegistry) -> None:
        self.projection_registry = projection_registry

    def plan_projection_chain(self, requested_projection: ProjectionDefinition) -> Sequence[ProjectionDefinition]:
        direct_projection = self.projection_registry.find_task_to_request_projection(
            source_view_kind=requested_projection.source_kind,
            target_request_kind=requested_projection.target_kind,
            dataset_id=requested_projection.dataset_scope,
        )
        return [direct_projection.describe_projection()]
