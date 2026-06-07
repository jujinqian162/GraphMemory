from __future__ import annotations

from graph_memory.models.graph_retriever.config.records import TrainableTrainingConfig
from graph_memory.registry.conversions import (
    dense_config_from_encoder_settings,
    trainable_training_config_from_trainer_settings,
)
from graph_memory.registry.retrieval import DenseEncoderSettings
from graph_memory.registry.training import RgcnTrainerSettings
from graph_memory.retrieval.methods.flat.dense import DenseConfig


def test_dense_config_from_encoder_settings_maps_all_runtime_fields() -> None:
    settings = DenseEncoderSettings(
        model_name="models/test-e5",
        query_prefix="q: ",
        passage_prefix="p: ",
        batch_size=17,
    )

    assert dense_config_from_encoder_settings(settings) == DenseConfig(
        model_name="models/test-e5",
        query_prefix="q: ",
        passage_prefix="p: ",
        batch_size=17,
    )


def test_trainable_training_config_from_trainer_settings_excludes_device() -> None:
    settings = RgcnTrainerSettings(
        optimizer_name="AdamW",
        learning_rate=0.02,
        batch_size=3,
        max_grad_norm=0.5,
        random_seed=23,
        pos_weight_enabled=True,
        epochs=4,
        device="cuda:0",
    )

    assert trainable_training_config_from_trainer_settings(settings) == TrainableTrainingConfig(
        optimizer_name="AdamW",
        learning_rate=0.02,
        batch_size=3,
        max_grad_norm=0.5,
        random_seed=23,
        pos_weight_enabled=True,
        epochs=4,
    )
