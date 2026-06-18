from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from abstraction.domain.retrieval.capabilities import MethodRegistry
from abstraction.domain.scripts.artifact_io import ScriptArtifactReader, ScriptArtifactWriter
from abstraction.domain.scripts.cli_arguments import ScriptCliArguments
from abstraction.domain.scripts.steps import (
    ArtifactPublicationStep,
    DatasetPreparationStep,
    EvaluationStep,
    GraphBuildStep,
    RequestProjectionStep,
    RetrievalStep,
    ScriptBoundaryReviewStep,
    TrainingStep,
)


@dataclass(frozen=True)
class PrepareDatasetScriptContext:
    dataset_preparation: DatasetPreparationStep
    artifact_writer: ScriptArtifactWriter


@dataclass(frozen=True)
class ProjectRequestScriptContext:
    artifact_reader: ScriptArtifactReader
    artifact_writer: ScriptArtifactWriter
    request_projection: RequestProjectionStep


@dataclass(frozen=True)
class BuildGraphScriptContext:
    artifact_reader: ScriptArtifactReader
    artifact_writer: ScriptArtifactWriter
    method_registry: MethodRegistry
    graph_build: GraphBuildStep


@dataclass(frozen=True)
class TrainMethodScriptContext:
    artifact_reader: ScriptArtifactReader
    artifact_writer: ScriptArtifactWriter
    method_registry: MethodRegistry
    training: TrainingStep


@dataclass(frozen=True)
class TuneMethodScriptContext:
    method_registry: MethodRegistry
    artifact_writer: ScriptArtifactWriter


@dataclass(frozen=True)
class RetrieveScriptContext:
    artifact_reader: ScriptArtifactReader
    artifact_writer: ScriptArtifactWriter
    method_registry: MethodRegistry
    retrieval: RetrievalStep


@dataclass(frozen=True)
class EvaluateScriptContext:
    artifact_reader: ScriptArtifactReader
    artifact_writer: ScriptArtifactWriter
    method_registry: MethodRegistry
    evaluation: EvaluationStep
    boundary_review: ScriptBoundaryReviewStep
    artifact_publication: ArtifactPublicationStep


class ScriptCompositionRoot(Protocol):
    def build_prepare_dataset_context(self, args: ScriptCliArguments) -> PrepareDatasetScriptContext:
        ...

    def build_project_request_context(self, args: ScriptCliArguments) -> ProjectRequestScriptContext:
        ...

    def build_graph_context(self, args: ScriptCliArguments) -> BuildGraphScriptContext:
        ...

    def build_train_context(self, args: ScriptCliArguments) -> TrainMethodScriptContext:
        ...

    def build_tune_context(self, args: ScriptCliArguments) -> TuneMethodScriptContext:
        ...

    def build_retrieve_context(self, args: ScriptCliArguments) -> RetrieveScriptContext:
        ...

    def build_evaluate_context(self, args: ScriptCliArguments) -> EvaluateScriptContext:
        ...


class ScriptLocalCompositionRoot:  # implement ScriptCompositionRoot
    def build_prepare_dataset_context(self, args: ScriptCliArguments) -> PrepareDatasetScriptContext:
        pass

    def build_project_request_context(self, args: ScriptCliArguments) -> ProjectRequestScriptContext:
        pass

    def build_graph_context(self, args: ScriptCliArguments) -> BuildGraphScriptContext:
        pass

    def build_train_context(self, args: ScriptCliArguments) -> TrainMethodScriptContext:
        pass

    def build_tune_context(self, args: ScriptCliArguments) -> TuneMethodScriptContext:
        pass

    def build_retrieve_context(self, args: ScriptCliArguments) -> RetrieveScriptContext:
        pass

    def build_evaluate_context(self, args: ScriptCliArguments) -> EvaluateScriptContext:
        pass
