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
    counts: dict[str, JsonValue]


class EvidenceGraphResult(_StageResult):
    split: SplitName
    artifact: EvidenceGraphArtifactRef
    statistics: dict[str, JsonValue]


class TrainingPairsResult(_StageResult):
    artifact: TrainingPairsArtifactRef
    summary: dict[str, JsonValue]


class FrozenEmbeddingsResult(_StageResult):
    family: str = Field(min_length=1)
    artifact: FrozenEmbeddingsArtifactRef
    embedding_dim: int = Field(gt=0)
    row_count: int = Field(gt=0)
    task_count: int = Field(gt=0)


class ModelResult(_StageResult):
    method: str = Field(min_length=1)
    artifact: ModelArtifactRef
    training_history: tuple[dict[str, JsonValue], ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RankingResult(_StageResult):
    method: str = Field(min_length=1)
    artifact: PredictionsArtifactRef
    provenance: dict[str, JsonValue]
    production_seconds: float = Field(ge=0.0)


class EvaluationResult(_StageResult):
    method: str = Field(min_length=1)
    artifact: EvaluationArtifactRef
    metric_rows: tuple[MetricRow, ...]
    per_task_rows: tuple[PerTaskMetricRow, ...] = ()
    failure_case_count: int = Field(ge=0)


__all__ = [
    "EvaluationResult",
    "EvidenceGraphResult",
    "FrozenEmbeddingsResult",
    "ModelResult",
    "PreparedSplitResult",
    "RankingResult",
    "TrainingPairsResult",
]
