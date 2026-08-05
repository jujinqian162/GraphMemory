from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from graph_memory.experiment.artifacts import (
    EvaluationArtifactRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
)
from graph_memory.evaluation.contracts import MetricRow, PerTaskMetricRow


class _StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


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
    "ModelResult",
    "RankingResult",
]
