from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graph_memory.experiment.artifacts import (
    ArtifactRef,
    EvaluationArtifactRef,
    ModelArtifactRef,
    PredictionsArtifactRef,
)


class _ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class FinalExperimentResult(_ResultModel):
    method: str = Field(min_length=1)
    variant: str | None
    ranking: PredictionsArtifactRef
    evaluation: EvaluationArtifactRef
    model: ModelArtifactRef | None = None
    dependency_models: tuple[ModelArtifactRef, ...] = ()
    assets: tuple[ArtifactRef, ...]


__all__ = ["FinalExperimentResult"]
