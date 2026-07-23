from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from graph_memory.models.provenance_rgcn.config import (
    PROVENANCE_RGCN_CHECKPOINT_FAMILY,
    PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION,
    ProvenanceRgcnModelConfig,
    ProvenanceRgcnTrainingConfig,
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
    payload: dict[str, Any] = {
        "checkpoint_family": PROVENANCE_RGCN_CHECKPOINT_FAMILY,
        "schema_version": PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION,
        "method_name": method_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict or {},
        "epoch": epoch,
        "global_step": global_step,
        "best_dev_metric": float(best_dev_metric),
        "best_metrics": dict(best_metrics or {"dev_joint": float(best_dev_metric)}),
        "effective_variant": effective_variant,
        "candidate_loss_protocol": "provenance-candidate-loss-v2",
        "candidate_loss_type": "task_balanced_pairwise_logistic",
        "batch_semantics": "disconnected_union_task_graphs",
        "selection_objective": (
            "0.50*full_support_at_5+0.25*mrr+0.25*edge_f1_at_10"
        ),
        "scientific_identity": dict(
            scientific_identity
            or {
                "dataset": "unspecified",
                "construction": "unspecified",
                "pairs": "unspecified",
                "encoder": model_config.encoder_model,
            }
        ),
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
    }
    _validate_payload(payload, expected_method=method_name)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return payload


def load_provenance_rgcn_checkpoint(
    path: str | Path,
    *,
    expected_method: str | None = "execution_provenance_rgcn_retriever",
    map_location: str | torch.device = "cpu",
) -> ProvenanceRgcnCheckpoint:
    value = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("Provenance R-GCN checkpoint must be a dictionary.")
    payload = cast(dict[str, Any], value)
    _validate_payload(payload, expected_method=expected_method)
    model_config_value = payload["model_config"]
    training_config_value = payload["training_config"]
    if not isinstance(model_config_value, dict) or not isinstance(
        training_config_value, dict
    ):
        raise ValueError("Provenance R-GCN checkpoint config payloads must be objects.")
    return ProvenanceRgcnCheckpoint(
        payload=payload,
        model_config=ProvenanceRgcnModelConfig.from_dict(model_config_value),
        training_config=ProvenanceRgcnTrainingConfig.from_dict(training_config_value),
    )


def _validate_payload(payload: dict[str, Any], *, expected_method: str | None) -> None:
    family = payload.get("checkpoint_family")
    if family != PROVENANCE_RGCN_CHECKPOINT_FAMILY:
        raise ValueError(
            "Provenance R-GCN checkpoint family mismatch: "
            f"expected={PROVENANCE_RGCN_CHECKPOINT_FAMILY!r} observed={family!r}."
        )
    observed_schema = payload.get("schema_version")
    if observed_schema != PROVENANCE_RGCN_CHECKPOINT_SCHEMA_VERSION:
        if observed_schema == 3:
            raise ValueError(
                "Legacy provenance R-GCN schema v3 does not use disconnected-union "
                "graph-batch semantics; retrain with the current runtime."
            )
        raise ValueError("Unsupported provenance R-GCN checkpoint schema version.")
    if expected_method is not None and payload.get("method_name") != expected_method:
        raise ValueError(
            "Provenance R-GCN checkpoint method mismatch: "
            f"expected={expected_method!r} observed={payload.get('method_name')!r}."
        )
    for field_name in (
        "model_state_dict",
        "model_config",
        "training_config",
        "global_step",
        "best_metrics",
        "effective_variant",
        "candidate_loss_protocol",
        "candidate_loss_type",
        "batch_semantics",
        "selection_objective",
        "scientific_identity",
    ):
        if field_name not in payload:
            raise ValueError(f"Provenance R-GCN checkpoint missing field={field_name}.")
    model_config = payload.get("model_config")
    required_model_fields = {
        "ablation_name",
        "message_transform_type",
        "edge_weight_policy",
        "structured_pool_size",
        "structured_seed_top_s",
        "preserve_node_top_n",
        "edge_accept_threshold",
        "feed_message_topology",
        "feed_message_shuffle_seed",
        "node_type_vocab",
        "relation_vocab",
    }
    if not isinstance(model_config, dict) or not required_model_fields <= set(
        model_config
    ):
        missing = sorted(
            required_model_fields
            - (set(model_config) if isinstance(model_config, dict) else set())
        )
        raise ValueError(
            "Provenance R-GCN checkpoint model_config is incomplete for schema v4: "
            f"missing={missing}."
        )
    training_config = payload.get("training_config")
    expected_training_fields = {
        "learning_rate",
        "per_device_graph_batch_size",
        "epochs",
        "max_grad_norm",
        "random_seed",
        "candidate_loss_weight",
        "edge_loss_weight",
    }
    if not isinstance(training_config, dict) or set(
        training_config
    ) != expected_training_fields:
        raise ValueError(
            "Provenance R-GCN checkpoint training_config uses legacy batch semantics "
            "or does not match schema v4."
        )
    graph_batch = training_config["per_device_graph_batch_size"]
    if not isinstance(graph_batch, int) or graph_batch <= 0:
        raise ValueError("Invalid provenance R-GCN graph-batch metadata.")
    if payload.get("candidate_loss_protocol") != "provenance-candidate-loss-v2":
        raise ValueError("Unsupported provenance candidate-loss protocol.")
    if payload.get("batch_semantics") != "disconnected_union_task_graphs":
        raise ValueError("Unsupported provenance graph-batch semantics.")
    scientific_identity = payload.get("scientific_identity")
    required_identity = {"dataset", "construction", "pairs", "encoder"}
    if not isinstance(scientific_identity, dict) or not required_identity <= set(
        scientific_identity
    ):
        raise ValueError(
            "Provenance R-GCN checkpoint scientific_identity is incomplete."
        )


__all__ = [
    "ProvenanceRgcnCheckpoint",
    "load_provenance_rgcn_checkpoint",
    "save_provenance_rgcn_checkpoint",
]
