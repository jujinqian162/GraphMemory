from __future__ import annotations

from typing import Protocol

from graph_memory.models.graph_retriever.config.records import RgcnTrainingConfig
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.retrieval.methods.flat.dense import DenseConfig


class TrainerSettingsLike(Protocol):
    @property
    def optimizer_name(self) -> str: ...

    @property
    def learning_rate(self) -> float: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def max_grad_norm(self) -> float: ...

    @property
    def random_seed(self) -> int: ...

    @property
    def pos_weight_enabled(self) -> bool: ...

    @property
    def epochs(self) -> int: ...


def dense_config_from_encoder_settings(settings: DenseEncoderSettings) -> DenseConfig:
    return DenseConfig(
        model_name=settings.model_name,
        query_prefix=settings.query_prefix,
        passage_prefix=settings.passage_prefix,
        batch_size=settings.batch_size,
    )


def rgcn_training_config_from_trainer_settings(settings: TrainerSettingsLike) -> RgcnTrainingConfig:
    return RgcnTrainingConfig(
        optimizer_name=settings.optimizer_name,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        max_grad_norm=settings.max_grad_norm,
        random_seed=settings.random_seed,
        pos_weight_enabled=settings.pos_weight_enabled,
        epochs=settings.epochs,
    )


__all__ = [
    "dense_config_from_encoder_settings",
    "rgcn_training_config_from_trainer_settings",
]
