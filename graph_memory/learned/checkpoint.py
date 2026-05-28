from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from graph_memory.observability import now_iso
from graph_memory.types import NodeFeatureConfig, TrainableModelConfig, TrainableTrainingConfig
from graph_memory.validation import validate_trainable_checkpoint_metadata


@dataclass(frozen=True)
class TrainableCheckpoint:
    """
    Loaded trainable checkpoint with parsed config objects.
    已加载并解析 config 对象的可训练 checkpoint。

    Fields / 字段:
    - payload: Raw PyTorch checkpoint dictionary.
      payload：原始 PyTorch checkpoint 字典。
    - model_config: Parsed model reconstruction config.
      model_config：解析后的模型重建配置。
    - training_config: Parsed training audit config.
      training_config：解析后的训练审计配置。
    """

    payload: dict[str, Any]
    model_config: TrainableModelConfig
    training_config: TrainableTrainingConfig


def save_trainable_checkpoint(
    path: str | Path,
    *,
    method_name: str,
    model: nn.Module,
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    epoch: int,
    global_step: int,
    best_dev_metric: float,
    model_config: TrainableModelConfig,
    training_config: TrainableTrainingConfig,
) -> dict[str, Any]:
    """
    Save one validated trainable checkpoint.
    保存一个已验证的可训练 checkpoint。
    """

    payload: dict[str, Any] = {
        "checkpoint_version": 1,
        "method_name": method_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": epoch,
        "global_step": global_step,
        "best_dev_metric": float(best_dev_metric),
        "model_config": model_config.to_json_dict(),
        "training_config": training_config.to_json_dict(),
        "created_at": now_iso(),
    }
    validate_trainable_checkpoint_metadata(payload, expected_method=method_name)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return payload


def load_trainable_checkpoint(
    path: str | Path,
    *,
    expected_method: str | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainableCheckpoint:
    """
    Load and validate one trainable checkpoint.
    加载并验证一个可训练 checkpoint。
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Trainable checkpoint must be a dictionary: {path}")
    typed_payload = cast(dict[str, Any], payload)
    validate_trainable_checkpoint_metadata(typed_payload, expected_method=expected_method)
    return TrainableCheckpoint(
        payload=typed_payload,
        model_config=model_config_from_record(typed_payload["model_config"]),
        training_config=training_config_from_record(typed_payload["training_config"]),
    )


def model_config_from_record(record: object) -> TrainableModelConfig:
    """
    Parse a trainable model config record into a dataclass.
    将 trainable model config 记录解析为 dataclass。
    """

    if not isinstance(record, dict):
        raise ValueError("model_config must be an object.")
    feature_config = record.get("feature_config")
    if not isinstance(feature_config, dict):
        raise ValueError("model_config.feature_config must be an object.")
    return TrainableModelConfig(
        method_name=str(record["method_name"]),
        encoder_model=str(record["encoder_model"]),
        encoder_dim=int(record["encoder_dim"]),
        query_prefix=str(record["query_prefix"]),
        passage_prefix=str(record["passage_prefix"]),
        hidden_dim=int(record["hidden_dim"]),
        num_layers=int(record["num_layers"]),
        dropout=float(record["dropout"]),
        feature_config=NodeFeatureConfig(
            node_feature_names=tuple(str(name) for name in feature_config["node_feature_names"]),
            scorer_feature_names=tuple(str(name) for name in feature_config["scorer_feature_names"]),
        ),
        relation_vocab=tuple(str(name) for name in record["relation_vocab"]),
        graph_encoder_type=str(record["graph_encoder_type"]),
        message_transform_type=str(record["message_transform_type"]),
        edge_weight_policy=str(record["edge_weight_policy"]),
        enabled_edge_types=tuple(str(edge_type) for edge_type in record["enabled_edge_types"]),
        ablation_name=str(record["ablation_name"]),
    )


def training_config_from_record(record: object) -> TrainableTrainingConfig:
    """
    Parse a trainable training config record into a dataclass.
    将 trainable training config 记录解析为 dataclass。
    """

    if not isinstance(record, dict):
        raise ValueError("training_config must be an object.")
    return TrainableTrainingConfig(
        optimizer_name=str(record["optimizer_name"]),
        learning_rate=float(record["learning_rate"]),
        batch_size=int(record["batch_size"]),
        max_grad_norm=float(record["max_grad_norm"]),
        random_seed=int(record["random_seed"]),
        pos_weight_enabled=bool(record["pos_weight_enabled"]),
        epochs=int(record["epochs"]),
    )
