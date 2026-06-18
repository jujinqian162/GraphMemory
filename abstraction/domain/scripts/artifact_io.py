from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.scripts.steps import (
    EvaluationResult,
    GraphBuildResult,
    PreparedDatasetResult,
    ProjectedRequestResult,
    RetrievalResult,
    TrainingResult,
)
from abstraction.domain.scripts.retrieval_artifacts import TemporalMemorySignals


class ScriptArtifactReader(Protocol):
    def load_prepared_dataset(self, artifact_refs: Sequence[str]) -> PreparedDatasetResult:
        ...

    def load_projected_request(self, artifact_refs: Sequence[str]) -> ProjectedRequestResult:
        ...

    def load_graph_result(self, artifact_refs: Sequence[str]) -> GraphBuildResult | None:
        ...

    def load_training_result(self, artifact_refs: Sequence[str]) -> TrainingResult | None:
        ...

    def load_required_training_result(self, artifact_refs: Sequence[str]) -> TrainingResult:
        ...

    def load_temporal_memory_signals(self, artifact_refs: Sequence[str]) -> TemporalMemorySignals:
        ...

    def load_retrieval_result(self, artifact_refs: Sequence[str]) -> RetrievalResult:
        ...

    def load_evaluation_result(self, artifact_refs: Sequence[str]) -> EvaluationResult:
        ...


class ScriptArtifactWriter(Protocol):
    def write_prepared_dataset(self, result: PreparedDatasetResult, artifact_refs: Sequence[str]) -> None:
        ...

    def write_projected_request(self, result: ProjectedRequestResult, artifact_refs: Sequence[str]) -> None:
        ...

    def write_graph_result(self, result: GraphBuildResult | None, artifact_refs: Sequence[str]) -> None:
        ...

    def write_training_result(self, result: TrainingResult | None, artifact_refs: Sequence[str]) -> None:
        ...

    def write_retrieval_result(self, result: RetrievalResult, artifact_refs: Sequence[str]) -> None:
        ...

    def write_evaluation_result(self, result: EvaluationResult, artifact_refs: Sequence[str]) -> None:
        ...


class ManifestScriptArtifactReader:  # implement ScriptArtifactReader
    def load_prepared_dataset(self, artifact_refs: Sequence[str]) -> PreparedDatasetResult:
        pass

    def load_projected_request(self, artifact_refs: Sequence[str]) -> ProjectedRequestResult:
        pass

    def load_graph_result(self, artifact_refs: Sequence[str]) -> GraphBuildResult | None:
        pass

    def load_training_result(self, artifact_refs: Sequence[str]) -> TrainingResult | None:
        pass

    def load_required_training_result(self, artifact_refs: Sequence[str]) -> TrainingResult:
        pass

    def load_temporal_memory_signals(self, artifact_refs: Sequence[str]) -> TemporalMemorySignals:
        pass

    def load_retrieval_result(self, artifact_refs: Sequence[str]) -> RetrievalResult:
        pass

    def load_evaluation_result(self, artifact_refs: Sequence[str]) -> EvaluationResult:
        pass


class ManifestScriptArtifactWriter:  # implement ScriptArtifactWriter
    def write_prepared_dataset(self, result: PreparedDatasetResult, artifact_refs: Sequence[str]) -> None:
        pass

    def write_projected_request(self, result: ProjectedRequestResult, artifact_refs: Sequence[str]) -> None:
        pass

    def write_graph_result(self, result: GraphBuildResult | None, artifact_refs: Sequence[str]) -> None:
        pass

    def write_training_result(self, result: TrainingResult | None, artifact_refs: Sequence[str]) -> None:
        pass

    def write_retrieval_result(self, result: RetrievalResult, artifact_refs: Sequence[str]) -> None:
        pass

    def write_evaluation_result(self, result: EvaluationResult, artifact_refs: Sequence[str]) -> None:
        pass
