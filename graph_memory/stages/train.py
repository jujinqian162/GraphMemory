from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias
from typing_extensions import assert_never

from graph_memory.models.dense_finetune.training import DenseFinetuneTrainingResult
from graph_memory.models.graph_retriever.training import RgcnTrainingResult
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import (
    DenseFinetuneTrainStageConfig,
    RgcnTrainStageConfig,
    TrainStageConfig,
)
from graph_memory.registry.training import TrainPayload

TrainingResult: TypeAlias = RgcnTrainingResult | DenseFinetuneTrainingResult


@dataclass(frozen=True)
class TrainStageResult:
    result: TrainingResult


def run_train_stage(
    config: TrainStageConfig,
    *,
    payload: TrainPayload,
) -> TrainStageResult:
    trainer = Registry.training.build(config.job)
    result = trainer.train(payload)
    if isinstance(config, RgcnTrainStageConfig):
        if not isinstance(result, RgcnTrainingResult):
            raise TypeError(f"R-GCN training returned {type(result).__name__}.")
        return TrainStageResult(result=result)
    if isinstance(config, DenseFinetuneTrainStageConfig):
        if not isinstance(result, DenseFinetuneTrainingResult):
            raise TypeError(f"Dense-FT training returned {type(result).__name__}.")
        return TrainStageResult(result=result)
    assert_never(config)


__all__ = ["TrainingResult", "TrainStageResult", "run_train_stage"]
