from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from graph_memory.experiment.artifacts import (
    DatasetArtifactRef,
    EvaluationArtifactRef,
    EvidenceGraphArtifactRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
    TrainingPairsArtifactRef,
)
from graph_memory.experiment.config import SplitName


class _StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class PreparedSplitResult(_StageResult):
    stage: Literal["prepare"] = "prepare"
    split: SplitName
    artifact: DatasetArtifactRef
    counts: dict[str, JsonValue]


class EvidenceGraphResult(_StageResult):
    stage: Literal["evidence_graphs"] = "evidence_graphs"
    split: SplitName
    artifact: EvidenceGraphArtifactRef
    statistics: dict[str, JsonValue]


class TrainingPairsResult(_StageResult):
    stage: Literal["pairs"] = "pairs"
    artifact: TrainingPairsArtifactRef
    summary: dict[str, JsonValue]


class ModelResult(_StageResult):
    stage: Literal["train"] = "train"
    method: str = Field(min_length=1)
    artifact: ModelArtifactRef
    training_history: tuple[dict[str, JsonValue], ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RankingResult(_StageResult):
    stage: Literal["rank"] = "rank"
    method: str = Field(min_length=1)
    artifact: PredictionsArtifactRef
    provenance: dict[str, JsonValue]
    production_seconds: float = Field(ge=0.0)


class EvaluationResult(_StageResult):
    stage: Literal["evaluate"] = "evaluate"
    method: str = Field(min_length=1)
    artifact: EvaluationArtifactRef
    metric_rows: tuple[dict[str, JsonValue], ...]
    per_task_rows: tuple[dict[str, JsonValue], ...] = ()
    failure_case_count: int = Field(ge=0)


class BenchmarkResult(_StageResult):
    stage: Literal["benchmark"] = "benchmark"
    warmup: int = Field(ge=0)
    repetitions: int = Field(gt=0)
    metrics: dict[str, float]


__all__ = [
    "BenchmarkResult",
    "EvaluationResult",
    "EvidenceGraphResult",
    "ModelResult",
    "PreparedSplitResult",
    "RankingResult",
    "TrainingPairsResult",
]
