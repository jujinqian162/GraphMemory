from __future__ import annotations

from dataclasses import dataclass

from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import TrainStageConfig
from graph_memory.registry.training import TrainPayload


@dataclass(frozen=True)
class TrainStageResult:
    result: object


def run_train_stage(
    config: TrainStageConfig,
    *,
    payload: TrainPayload,
) -> TrainStageResult:
    trainer = Registry.training.build(config.job)
    return TrainStageResult(result=trainer.train(payload))


__all__ = ["TrainStageResult", "run_train_stage"]
