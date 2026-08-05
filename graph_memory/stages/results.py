from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from graph_memory.experiment.artifacts import (
    DatasetArtifactRef,
    EvaluationArtifactRef,
    EvidenceGraphArtifactRef,
    FrozenEmbeddingsArtifactRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
    TrainingPairsArtifactRef,
)
from graph_memory.evaluation.contracts import MetricRow, PerTaskMetricRow
from graph_memory.experiment.config import SplitName


class _StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class PreparedSplitResult(_StageResult):
    split: SplitName
    artifact: DatasetArtifactRef


class EvidenceGraphResult(_StageResult):
    artifact: EvidenceGraphArtifactRef


class TrainingPairsResult(_StageResult):
    artifact: TrainingPairsArtifactRef


class FrozenEmbeddingsResult(_StageResult):
    artifact: FrozenEmbeddingsArtifactRef


class ModelResult(_StageResult):
    artifact: ModelArtifactRef
    training_history: tuple[dict[str, JsonValue], ...] = ()


class RankingResult(_StageResult):
    artifact: PredictionsArtifactRef
    production_seconds: float = Field(ge=0.0)


class EvaluationResult(_StageResult):
    artifact: EvaluationArtifactRef
    metric_rows: tuple[MetricRow, ...]
    per_task_rows: tuple[PerTaskMetricRow, ...] = ()


__all__ = [
    "EvaluationResult",
    "EvidenceGraphResult",
    "FrozenEmbeddingsResult",
    "ModelResult",
    "PreparedSplitResult",
    "RankingResult",
    "TrainingPairsResult",
]
