from __future__ import annotations

from graph_memory.contracts.common import ALLOWED_EDGE_TYPES
from graph_memory.validation.common import (
    ContractValidationError,
    _reject_unknown_fields,
    _require_record,
    _required_finite_number,
    _required_int,
    _required_string,
    _to_plain_dict,
    _validate_string_sequence,
)

NODE_FEATURE_CONFIG_FIELDS = {"node_feature_names", "scorer_feature_names"}
RGCN_MODEL_CONFIG_FIELDS = {
    "method_name",
    "encoder_model",
    "encoder_dim",
    "query_prefix",
    "passage_prefix",
    "encoder_batch_size",
    "hidden_dim",
    "num_layers",
    "dropout",
    "feature_config",
    "relation_vocab",
    "graph_encoder_type",
    "message_transform_type",
    "edge_weight_policy",
    "enabled_edge_types",
    "ablation_name",
}
RGCN_TRAINING_CONFIG_FIELDS = {
    "optimizer_name",
    "learning_rate",
    "batch_size",
    "max_grad_norm",
    "random_seed",
    "pos_weight_enabled",
    "epochs",
}
RGCN_CHECKPOINT_FIELDS = {
    "schema_version",
    "method_name",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "global_step",
    "best_dev_metric",
    "model_config",
    "training_config",
    "created_at",
}
KNOWN_NODE_FEATURES = {"seed_score", "seed_rank_percentile", "is_question_node"}


def validate_rgcn_model_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    _reject_unknown_fields(config_dict, RGCN_MODEL_CONFIG_FIELDS, "R-GCN model config")
    _required_string(config_dict, "method_name", "R-GCN model config")
    _required_string(config_dict, "encoder_model", "R-GCN model config")
    _required_int(config_dict, "encoder_dim", "R-GCN model config", minimum=1)
    _required_string(config_dict, "query_prefix", "R-GCN model config")
    _required_string(config_dict, "passage_prefix", "R-GCN model config")
    _required_int(config_dict, "encoder_batch_size", "R-GCN model config", minimum=1)
    _required_int(config_dict, "hidden_dim", "R-GCN model config", minimum=1)
    _required_int(config_dict, "num_layers", "R-GCN model config", minimum=0)
    dropout = _required_finite_number(
        config_dict, "dropout", "R-GCN model config", minimum=0.0
    )
    if dropout >= 1.0:
        raise ContractValidationError(
            "Invalid R-GCN model config: dropout must be < 1.0."
        )

    _validate_node_feature_config(config_dict.get("feature_config"))
    _validate_string_sequence(
        config_dict.get("relation_vocab"), "relation_vocab", allow_empty=False
    )
    graph_encoder_type = _required_string(
        config_dict, "graph_encoder_type", "R-GCN model config"
    )
    if graph_encoder_type not in {"identity", "rgcn"}:
        raise ContractValidationError(
            "Invalid R-GCN model config: graph_encoder_type must be identity or rgcn."
        )
    message_transform_type = _required_string(
        config_dict, "message_transform_type", "R-GCN model config"
    )
    if message_transform_type not in {"typed", "shared"}:
        raise ContractValidationError(
            "Invalid R-GCN model config: message_transform_type must be typed or shared."
        )
    edge_weight_policy = _required_string(
        config_dict, "edge_weight_policy", "R-GCN model config"
    )
    if edge_weight_policy not in {"artifact", "uniform"}:
        raise ContractValidationError(
            "Invalid R-GCN model config: edge_weight_policy must be artifact or uniform."
        )
    enabled_edge_types = set(
        _validate_string_sequence(
            config_dict.get("enabled_edge_types"),
            "enabled_edge_types",
            allow_empty=True,
        )
    )
    unknown_edge_types = sorted(enabled_edge_types - ALLOWED_EDGE_TYPES)
    if unknown_edge_types:
        raise ContractValidationError(
            f"Invalid R-GCN model config: unsupported enabled_edge_types={unknown_edge_types}."
        )
    _required_string(config_dict, "ablation_name", "R-GCN model config")


def validate_rgcn_training_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    _reject_unknown_fields(
        config_dict, RGCN_TRAINING_CONFIG_FIELDS, "R-GCN training config"
    )
    if (
        _required_string(config_dict, "optimizer_name", "R-GCN training config")
        != "AdamW"
    ):
        raise ContractValidationError(
            "Invalid R-GCN training config: optimizer_name must be AdamW."
        )
    _required_finite_number(
        config_dict, "learning_rate", "R-GCN training config", minimum=0.0
    )
    _required_int(config_dict, "batch_size", "R-GCN training config", minimum=1)
    _required_finite_number(
        config_dict, "max_grad_norm", "R-GCN training config", minimum=0.0
    )
    _required_int(config_dict, "random_seed", "R-GCN training config")
    if not isinstance(config_dict.get("pos_weight_enabled"), bool):
        raise ContractValidationError(
            "Invalid R-GCN training config: pos_weight_enabled must be boolean."
        )
    _required_int(config_dict, "epochs", "R-GCN training config", minimum=1)


def validate_rgcn_checkpoint_metadata(
    checkpoint: object, *, expected_method: str | None = None
) -> None:
    checkpoint = _require_record(checkpoint, "R-GCN checkpoint")
    _reject_unknown_fields(checkpoint, RGCN_CHECKPOINT_FIELDS, "R-GCN checkpoint")
    schema_version = checkpoint.get("schema_version")
    if schema_version != 2:
        raise ContractValidationError(
            "Incompatible R-GCN checkpoint schema; retrain the node-wise model."
        )
    method_name = _required_string(checkpoint, "method_name", "R-GCN checkpoint")
    if expected_method is not None and method_name != expected_method:
        raise ContractValidationError(
            f"Invalid R-GCN checkpoint: method_name={method_name} does not match expected_method={expected_method}."
        )
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ContractValidationError(
            "Invalid R-GCN checkpoint: model_state_dict must be present."
        )
    if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
        raise ContractValidationError(
            "Invalid R-GCN checkpoint: optimizer_state_dict must be present."
        )
    if not isinstance(checkpoint.get("scheduler_state_dict"), dict):
        raise ContractValidationError(
            "Invalid R-GCN checkpoint: scheduler_state_dict must be present."
        )
    _required_int(checkpoint, "epoch", "R-GCN checkpoint", minimum=0)
    _required_int(checkpoint, "global_step", "R-GCN checkpoint", minimum=0)
    _required_finite_number(checkpoint, "best_dev_metric", "R-GCN checkpoint")
    validate_rgcn_model_config(checkpoint.get("model_config"))
    validate_rgcn_training_config(checkpoint.get("training_config"))
    _required_string(checkpoint, "created_at", "R-GCN checkpoint")


def _validate_node_feature_config(value: object) -> None:
    feature_config = _to_plain_dict(value)
    _reject_unknown_fields(
        feature_config, NODE_FEATURE_CONFIG_FIELDS, "node feature config"
    )
    for field_name in ["node_feature_names", "scorer_feature_names"]:
        feature_names = _validate_string_sequence(
            feature_config.get(field_name), field_name, allow_empty=True
        )
        unknown = sorted(set(feature_names) - KNOWN_NODE_FEATURES)
        if unknown:
            raise ContractValidationError(
                f"Invalid node feature config: unsupported {field_name}={unknown}."
            )


__all__ = [
    "validate_rgcn_checkpoint_metadata",
    "validate_rgcn_model_config",
    "validate_rgcn_training_config",
]
