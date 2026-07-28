from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, cast

import torch
from pydantic import Field, JsonValue, SkipValidation, model_validator
from torch import nn

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
    NonNegativeInt,
)
from graph_memory.models.provenance_rgcn.config import (
    PROVENANCE_RGCN_CHECKPOINT_FAMILY,
    PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION,
    ProvenanceRgcnModelConfig,
    ProvenanceRgcnTrainingConfig,
)
from graph_memory.retrieval.methods.ids import RetrievalMethodId

OpaqueState = SkipValidation[dict[str, object]]


class ProvenanceScientificIdentity(DomainModel):
    dataset: JsonValue
    construction: JsonValue
    pairs: JsonValue
    encoder: NonEmptyStr


class ProvenanceRgcnCheckpointEnvelope(DomainModel):
    checkpoint_family: Literal["execution_provenance_rgcn"]
    schema_version: Literal[4]
    method_name: RetrievalMethodId
    model_state_dict: OpaqueState
    optimizer_state_dict: OpaqueState
    epoch: NonNegativeInt
    global_step: NonNegativeInt
    best_dev_metric: FiniteFloat
    best_metrics: dict[NonEmptyStr, FiniteFloat]
    effective_variant: NonEmptyStr
    candidate_loss_protocol: Literal["provenance-candidate-loss-v2"]
    candidate_loss_type: Literal["task_balanced_pairwise_logistic"]
    batch_semantics: Literal["disconnected_union_task_graphs"]
    selection_objective: Literal[
        "0.50*full_support_at_5+0.25*mrr+0.25*edge_f1_at_10"
    ]
    scientific_identity: ProvenanceScientificIdentity
    checkpoint_model_config: ProvenanceRgcnModelConfig = Field(alias="model_config")
    training_config: ProvenanceRgcnTrainingConfig

    @model_validator(mode="before")
    @classmethod
    def _require_explicit_checkpoint_identity(cls, value: object) -> object:
        if not isinstance(value, Mapping) or value.get("schema_version") != 4:
            return value
        raw_config = value.get("model_config")
        if not isinstance(raw_config, Mapping):
            return value
        required = {
            name
            for name, field in ProvenanceRgcnModelConfig.model_fields.items()
            if isinstance(field.json_schema_extra, dict)
            and field.json_schema_extra.get("checkpoint_required") is True
        }
        missing = sorted(required - set(raw_config))
        if missing:
            raise ValueError(
                f"model_config is incomplete for schema v4; missing={missing}"
            )
        return value

    def require_method(self, expected_method: RetrievalMethodId | None) -> None:
        if expected_method is not None and self.method_name != expected_method:
            raise ValueError(
                f"checkpoint method_name={self.method_name} does not match "
                f"expected_method={expected_method}"
            )


@dataclass(frozen=True)
class ProvenanceRgcnCheckpoint:
    payload: dict[str, Any]
    model_config: ProvenanceRgcnModelConfig
    training_config: ProvenanceRgcnTrainingConfig


def save_provenance_rgcn_checkpoint(
    path: str | Path,
    *,
    method_name: str,
    model: nn.Module,
    model_config: ProvenanceRgcnModelConfig,
    training_config: ProvenanceRgcnTrainingConfig,
    optimizer_state_dict: dict[str, Any] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_dev_metric: float = 0.0,
    effective_variant: str = "full_rgcn",
    scientific_identity: Mapping[str, Any] | None = None,
    best_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    method_id = RetrievalMethodId(method_name)
    envelope = ProvenanceRgcnCheckpointEnvelope(
        checkpoint_family=PROVENANCE_RGCN_CHECKPOINT_FAMILY,
        schema_version=PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION,
        method_name=method_id,
        model_state_dict=cast(dict[str, object], dict(model.state_dict())),
        optimizer_state_dict=cast(
            dict[str, object], optimizer_state_dict or {}
        ),
        epoch=epoch,
        global_step=global_step,
        best_dev_metric=float(best_dev_metric),
        best_metrics=dict(best_metrics or {"dev_joint": float(best_dev_metric)}),
        effective_variant=effective_variant,
        candidate_loss_protocol="provenance-candidate-loss-v2",
        candidate_loss_type="task_balanced_pairwise_logistic",
        batch_semantics="disconnected_union_task_graphs",
        selection_objective=(
            "0.50*full_support_at_5+0.25*mrr+0.25*edge_f1_at_10"
        ),
        scientific_identity=ProvenanceScientificIdentity.model_validate(
            dict(scientific_identity)
            if scientific_identity is not None
            else {
                "dataset": "unspecified",
                "construction": "unspecified",
                "pairs": "unspecified",
                "encoder": model_config.encoder_model,
            }
        ),
        model_config=model_config,
        training_config=training_config,
    )
    envelope.require_method(method_id)
    payload = cast(
        dict[str, Any], envelope.model_dump(mode="python", by_alias=True)
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def load_provenance_rgcn_checkpoint(
    path: str | Path,
    *,
    expected_method: RetrievalMethodId | None = (
        RetrievalMethodId.EXECUTION_PROVENANCE_RGCN_RETRIEVER
    ),
    map_location: str | torch.device = "cpu",
) -> ProvenanceRgcnCheckpoint:
    value = torch.load(Path(path), map_location=map_location, weights_only=False)
    envelope = ProvenanceRgcnCheckpointEnvelope.model_validate(value)
    envelope.require_method(expected_method)
    payload = cast(
        dict[str, Any], envelope.model_dump(mode="python", by_alias=True)
    )
    return ProvenanceRgcnCheckpoint(
        payload=payload,
        model_config=envelope.checkpoint_model_config,
        training_config=envelope.training_config,
    )


__all__ = [
    "ProvenanceRgcnCheckpoint",
    "ProvenanceRgcnCheckpointEnvelope",
    "load_provenance_rgcn_checkpoint",
    "save_provenance_rgcn_checkpoint",
]
