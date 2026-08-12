from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import Field, SkipValidation, model_validator
from torch import nn

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeInt,
)
from graph_memory.models.graph_retriever.config.records import (
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.retrieval.methods.ids import RetrievalMethodId

RGCN_CHECKPOINT_SCHEMA_VERSION = 5
OpaqueState = SkipValidation[dict[str, object]]


class RgcnCheckpointEnvelope(DomainModel):
    schema_version: Literal[5]
    method_name: RetrievalMethodId
    model_state_dict: OpaqueState
    epoch: NonNegativeInt
    global_step: NonNegativeInt
    best_dev_metric: FiniteFloat
    checkpoint_model_config: RgcnModelConfig = Field(alias="model_config")
    training_config: RgcnTrainingConfig
    created_at: NonEmptyStr

    @model_validator(mode="after")
    def _validate_method_config(self) -> "RgcnCheckpointEnvelope":
        if self.checkpoint_model_config.method_name != self.method_name.value:
            raise ValueError(
                "checkpoint method_name and model_config.method_name must match"
            )
        return self

    def require_method(self, expected_method: RetrievalMethodId | None) -> None:
        if expected_method is not None and self.method_name != expected_method:
            raise ValueError(
                f"checkpoint method_name={self.method_name} does not match "
                f"expected_method={expected_method}"
            )


@dataclass(frozen=True)
class RgcnCheckpoint:
    payload: dict[str, Any]
    model_config: RgcnModelConfig
    training_config: RgcnTrainingConfig


def save_rgcn_checkpoint(
    path: str | Path,
    *,
    method_name: str,
    model: nn.Module,
    epoch: int,
    global_step: int,
    best_dev_metric: float,
    model_config: RgcnModelConfig,
    training_config: RgcnTrainingConfig,
) -> dict[str, Any]:
    method_id = RetrievalMethodId(method_name)
    envelope = RgcnCheckpointEnvelope(
        schema_version=RGCN_CHECKPOINT_SCHEMA_VERSION,
        method_name=method_id,
        model_state_dict=cast(dict[str, object], dict(model.state_dict())),
        epoch=epoch,
        global_step=global_step,
        best_dev_metric=float(best_dev_metric),
        model_config=model_config,
        training_config=training_config,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    envelope.require_method(method_id)
    payload = cast(
        dict[str, Any], envelope.model_dump(mode="python", by_alias=True)
    )
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return payload


def load_rgcn_checkpoint(
    path: str | Path,
    *,
    expected_method: RetrievalMethodId | None = None,
    map_location: str | torch.device,
) -> RgcnCheckpoint:
    value = torch.load(Path(path), map_location=map_location, weights_only=False)
    envelope = RgcnCheckpointEnvelope.model_validate(value)
    envelope.require_method(expected_method)
    payload = cast(
        dict[str, Any], envelope.model_dump(mode="python", by_alias=True)
    )
    return RgcnCheckpoint(
        payload=payload,
        model_config=envelope.checkpoint_model_config,
        training_config=envelope.training_config,
    )


__all__ = [
    "RGCN_CHECKPOINT_SCHEMA_VERSION",
    "RgcnCheckpoint",
    "RgcnCheckpointEnvelope",
    "load_rgcn_checkpoint",
    "save_rgcn_checkpoint",
]
