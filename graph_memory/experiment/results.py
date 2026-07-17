from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graph_memory.experiment.artifacts import ArtifactRef
from graph_memory.stages.results import (
    BenchmarkResult,
    EvaluationResult,
    ModelResult,
    RankingResult,
)


class _ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class FinalExperimentResult(_ResultModel):
    method: str = Field(min_length=1)
    variant: str | None
    ranking: RankingResult
    evaluation: EvaluationResult
    model: ModelResult | None = None
    dependency_models: tuple[ModelResult, ...] = ()
    benchmark: BenchmarkResult | None = None
    assets: tuple[ArtifactRef, ...]
    run_output: str | None = None


__all__ = ["FinalExperimentResult"]
