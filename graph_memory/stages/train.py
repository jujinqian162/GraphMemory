from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias
from typing_extensions import assert_never

from graph_memory.models.dense_finetune.training import DenseFinetuneTrainingResult
from graph_memory.models.graph_retriever.training import RgcnTrainingResult
from graph_memory.experiment.stage_models import (
    DenseFinetuneTrainStageConfig,
    OrdinaryRgcnTrainStageConfig,
    SeededRgcnTrainStageConfig,
    TrainStageConfig,
    ProvenanceRgcnTrainStageConfig,
)
from graph_memory.stages.train_payloads import TrainPayload
from graph_memory.stages.trainers import (
    DenseFinetuneMethodTrainer,
    RgcnGraphRetrieverTrainer,
    ProvenanceRgcnMethodTrainer,
)
from graph_memory.models.provenance_rgcn.training import ProvenanceTrainingResult

TrainingResult: TypeAlias = (
    RgcnTrainingResult | DenseFinetuneTrainingResult | ProvenanceTrainingResult
)


@dataclass(frozen=True)
class TrainStageResult:
    result: TrainingResult


def run_train_stage(
    config: TrainStageConfig,
    *,
    payload: TrainPayload,
) -> TrainStageResult:
    if isinstance(config, (OrdinaryRgcnTrainStageConfig, SeededRgcnTrainStageConfig)):
        result = RgcnGraphRetrieverTrainer(config).train(payload)
        if not isinstance(result, RgcnTrainingResult):
            raise TypeError(f"R-GCN training returned {type(result).__name__}.")
        return TrainStageResult(result=result)
    if isinstance(config, DenseFinetuneTrainStageConfig):
        result = DenseFinetuneMethodTrainer(config).train(payload)
        if not isinstance(result, DenseFinetuneTrainingResult):
            raise TypeError(f"Dense-FT training returned {type(result).__name__}.")
        return TrainStageResult(result=result)
    if isinstance(config, ProvenanceRgcnTrainStageConfig):
        result = ProvenanceRgcnMethodTrainer(config).train(payload)
        if not isinstance(result, ProvenanceTrainingResult):
            raise TypeError(
                f"Provenance R-GCN training returned {type(result).__name__}."
            )
        return TrainStageResult(result=result)
    assert_never(config)


__all__ = ["TrainingResult", "TrainStageResult", "run_train_stage"]
