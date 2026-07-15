from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import TypeAdapter
from torch import nn

from graph_memory.models.graph_retriever.config.records import (
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.validation import validate_rgcn_checkpoint_metadata

RGCN_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RgcnCheckpoint:
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
    model_config: RgcnModelConfig
    training_config: RgcnTrainingConfig


def save_rgcn_checkpoint(
    path: str | Path,
    *,
    method_name: str,
    model: nn.Module,
    optimizer_state_dict: dict[str, Any],
    scheduler_state_dict: dict[str, Any],
    epoch: int,
    global_step: int,
    best_dev_metric: float,
    model_config: RgcnModelConfig,
    training_config: RgcnTrainingConfig,
) -> dict[str, Any]:
    """
    Save one validated trainable checkpoint.
    保存一个已验证的可训练 checkpoint。
    """

    payload: dict[str, Any] = {
        "schema_version": RGCN_CHECKPOINT_SCHEMA_VERSION,
        "method_name": method_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": epoch,
        "global_step": global_step,
        "best_dev_metric": float(best_dev_metric),
        "model_config": model_config.to_json_dict(),
        "training_config": training_config.to_json_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_rgcn_checkpoint_metadata(payload, expected_method=method_name)
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return payload


def load_rgcn_checkpoint(
    path: str | Path,
    *,
    expected_method: str | None = None,
    map_location: str | torch.device = "cpu",
) -> RgcnCheckpoint:
    """
    Load and validate one trainable checkpoint.
    加载并验证一个可训练 checkpoint。
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"R-GCN checkpoint must be a dictionary: {path}")
    typed_payload = cast(dict[str, Any], payload)
    validate_rgcn_checkpoint_metadata(typed_payload, expected_method=expected_method)
    return RgcnCheckpoint(
        payload=typed_payload,
        model_config=TypeAdapter(RgcnModelConfig).validate_python(
            typed_payload["model_config"]
        ),
        training_config=TypeAdapter(RgcnTrainingConfig).validate_python(
            typed_payload["training_config"]
        ),
    )
