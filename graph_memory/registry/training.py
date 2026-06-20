from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from graph_memory.registry.method_configs import DenseFinetuneMethodSettings, RgcnMethodSettings, TrainJobSettings

if TYPE_CHECKING:
    from graph_memory.stages.train_payloads import TrainPayload


class TrainMethodTrainer(Protocol):
    def train(self, payload: "TrainPayload") -> object:
        ...


@dataclass(frozen=True)
class TrainingBuilderSpec:
    settings_type: type[object]
    build: Callable[[TrainJobSettings], TrainMethodTrainer]


@dataclass(frozen=True)
class TrainingRegistry:
    builders: dict[type[object], TrainingBuilderSpec]

    def build(self, settings: TrainJobSettings) -> TrainMethodTrainer:
        try:
            spec = self.builders[type(settings)]
        except KeyError as error:
            raise ValueError(f"Unsupported train settings type: {type(settings).__name__}") from error
        return spec.build(settings)


def build_training_registry() -> TrainingRegistry:
    from graph_memory.stages.trainers import DenseFinetuneMethodTrainer, RgcnGraphRetrieverTrainer

    return TrainingRegistry(
        builders={
            RgcnMethodSettings: TrainingBuilderSpec(
                RgcnMethodSettings,
                lambda settings: RgcnGraphRetrieverTrainer(cast(RgcnMethodSettings, settings)),
            ),
            DenseFinetuneMethodSettings: TrainingBuilderSpec(
                DenseFinetuneMethodSettings,
                lambda settings: DenseFinetuneMethodTrainer(cast(DenseFinetuneMethodSettings, settings)),
            ),
        }
    )


__all__ = [
    "TrainingRegistry",
    "build_training_registry",
]
