from __future__ import annotations

import math

from graph_memory.contracts.common import ALLOWED_EDGE_TYPES, NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES
from graph_memory.validation.common import (
    ContractValidationError,
    _reject_unknown_fields,
    _require_record,
    _require_tensor_1d,
    _require_tensor_2d,
    _required_attr,
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
    dropout = _required_finite_number(config_dict, "dropout", "R-GCN model config", minimum=0.0)
    if dropout >= 1.0:
        raise ContractValidationError("Invalid R-GCN model config: dropout must be < 1.0.")

    _validate_node_feature_config(config_dict.get("feature_config"))
    _validate_string_sequence(config_dict.get("relation_vocab"), "relation_vocab", allow_empty=False)
    graph_encoder_type = _required_string(config_dict, "graph_encoder_type", "R-GCN model config")
    if graph_encoder_type not in {"identity", "rgcn"}:
        raise ContractValidationError("Invalid R-GCN model config: graph_encoder_type must be identity or rgcn.")
    message_transform_type = _required_string(config_dict, "message_transform_type", "R-GCN model config")
    if message_transform_type not in {"typed", "shared"}:
        raise ContractValidationError("Invalid R-GCN model config: message_transform_type must be typed or shared.")
    edge_weight_policy = _required_string(config_dict, "edge_weight_policy", "R-GCN model config")
    if edge_weight_policy not in {"artifact", "uniform"}:
        raise ContractValidationError("Invalid R-GCN model config: edge_weight_policy must be artifact or uniform.")
    enabled_edge_types = set(_validate_string_sequence(config_dict.get("enabled_edge_types"), "enabled_edge_types", allow_empty=True))
    unknown_edge_types = sorted(enabled_edge_types - ALLOWED_EDGE_TYPES)
    if unknown_edge_types:
        raise ContractValidationError(
            f"Invalid R-GCN model config: unsupported enabled_edge_types={unknown_edge_types}."
        )
    _required_string(config_dict, "ablation_name", "R-GCN model config")


def validate_rgcn_training_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    _reject_unknown_fields(config_dict, RGCN_TRAINING_CONFIG_FIELDS, "R-GCN training config")
    if _required_string(config_dict, "optimizer_name", "R-GCN training config") != "AdamW":
        raise ContractValidationError("Invalid R-GCN training config: optimizer_name must be AdamW.")
    _required_finite_number(config_dict, "learning_rate", "R-GCN training config", minimum=0.0)
    _required_int(config_dict, "batch_size", "R-GCN training config", minimum=1)
    _required_finite_number(config_dict, "max_grad_norm", "R-GCN training config", minimum=0.0)
    _required_int(config_dict, "random_seed", "R-GCN training config")
    if not isinstance(config_dict.get("pos_weight_enabled"), bool):
        raise ContractValidationError("Invalid R-GCN training config: pos_weight_enabled must be boolean.")
    _required_int(config_dict, "epochs", "R-GCN training config", minimum=1)


def validate_rgcn_checkpoint_metadata(checkpoint: object, *, expected_method: str | None = None) -> None:
    checkpoint = _require_record(checkpoint, "R-GCN checkpoint")
    _reject_unknown_fields(checkpoint, RGCN_CHECKPOINT_FIELDS, "R-GCN checkpoint")
    method_name = _required_string(checkpoint, "method_name", "R-GCN checkpoint")
    if expected_method is not None and method_name != expected_method:
        raise ContractValidationError(
            f"Invalid R-GCN checkpoint: method_name={method_name} does not match expected_method={expected_method}."
        )
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ContractValidationError("Invalid R-GCN checkpoint: model_state_dict must be present.")
    if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
        raise ContractValidationError("Invalid R-GCN checkpoint: optimizer_state_dict must be present.")
    if not isinstance(checkpoint.get("scheduler_state_dict"), dict):
        raise ContractValidationError("Invalid R-GCN checkpoint: scheduler_state_dict must be present.")
    _required_int(checkpoint, "epoch", "R-GCN checkpoint", minimum=0)
    _required_int(checkpoint, "global_step", "R-GCN checkpoint", minimum=0)
    _required_finite_number(checkpoint, "best_dev_metric", "R-GCN checkpoint")
    validate_rgcn_model_config(checkpoint.get("model_config"))
    validate_rgcn_training_config(checkpoint.get("training_config"))
    _required_string(checkpoint, "created_at", "R-GCN checkpoint")


def validate_graph_batch(batch: object) -> None:
    total_nodes = _require_tensor_2d(batch, "node_embeddings", "graph batch").shape[0]
    node_features_shape = _require_tensor_2d(batch, "node_features", "graph batch").shape
    if node_features_shape[0] != total_nodes:
        raise ContractValidationError("Invalid graph batch: node_features first dimension must match node_embeddings.")
    edge_index_shape = _require_tensor_2d(batch, "edge_index", "graph batch").shape
    if edge_index_shape[0] != 2:
        raise ContractValidationError("Invalid graph batch: edge_index must have shape [2, num_message_edges].")
    relation_ids_shape = _require_tensor_1d(batch, "relation_ids", "graph batch").shape
    edge_weights_shape = _require_tensor_1d(batch, "edge_weights", "graph batch").shape
    if relation_ids_shape[0] != edge_index_shape[1]:
        raise ContractValidationError("Invalid graph batch: relation_ids length must match edge_index columns.")
    if edge_weights_shape[0] != edge_index_shape[1]:
        raise ContractValidationError("Invalid graph batch: edge_weights length must match edge_index columns.")

    query_indices_shape = _require_tensor_1d(batch, "query_node_indices", "graph batch").shape
    task_ids = _required_attr(batch, "task_ids", "graph batch")
    task_node_offsets = _required_attr(batch, "task_node_offsets", "graph batch")
    node_ids_by_task = _required_attr(batch, "node_ids_by_task", "graph batch")
    if not isinstance(task_ids, list) or not all(isinstance(task_id, str) and task_id for task_id in task_ids):
        raise ContractValidationError("Invalid graph batch: task_ids must be a list of non-empty strings.")
    if query_indices_shape[0] != len(task_ids):
        raise ContractValidationError("Invalid graph batch: query_node_indices length must match task_ids.")
    if not isinstance(task_node_offsets, list) or len(task_node_offsets) != len(task_ids) + 1:
        raise ContractValidationError("Invalid graph batch: task_node_offsets length must be len(task_ids) + 1.")
    if task_node_offsets[0] != 0 or task_node_offsets[-1] != total_nodes:
        raise ContractValidationError("Invalid graph batch: task_node_offsets must start at 0 and end at total_nodes.")
    if any(not isinstance(offset, int) for offset in task_node_offsets):
        raise ContractValidationError("Invalid graph batch: task_node_offsets entries must be integers.")
    if any(left > right for left, right in zip(task_node_offsets, task_node_offsets[1:])):
        raise ContractValidationError("Invalid graph batch: task_node_offsets must be monotonic.")
    if not isinstance(node_ids_by_task, list) or len(node_ids_by_task) != len(task_ids):
        raise ContractValidationError("Invalid graph batch: node_ids_by_task length must match task_ids.")
    for index, node_ids in enumerate(node_ids_by_task):
        if not isinstance(node_ids, list) or "q" not in node_ids:
            raise ContractValidationError("Invalid graph batch: every node_ids_by_task entry must be a list containing q.")
        expected_length = task_node_offsets[index + 1] - task_node_offsets[index]
        if len(node_ids) != expected_length:
            raise ContractValidationError("Invalid graph batch: node_ids_by_task lengths must match task_node_offsets.")


def validate_training_batch(batch: object) -> None:
    graph_batch = _required_attr(batch, "graph_batch", "training batch")
    validate_graph_batch(graph_batch)
    num_samples = _require_tensor_1d(batch, "sample_node_indices", "training batch").shape[0]
    if _require_tensor_1d(batch, "sample_query_indices", "training batch").shape[0] != num_samples:
        raise ContractValidationError("Invalid training batch: sample_query_indices length must match samples.")
    if _require_tensor_1d(batch, "labels", "training batch").shape[0] != num_samples:
        raise ContractValidationError("Invalid training batch: labels length must match samples.")
    sample_node_features_shape = _require_tensor_2d(batch, "sample_node_features", "training batch").shape
    if sample_node_features_shape[0] != num_samples:
        raise ContractValidationError("Invalid training batch: sample_node_features first dimension must match samples.")
    sample_node_ids: list[object] | None = None
    for field_name in ["sample_task_ids", "sample_node_ids", "sample_types"]:
        value = _required_attr(batch, field_name, "training batch")
        if not isinstance(value, list) or len(value) != num_samples:
            raise ContractValidationError(f"Invalid training batch: {field_name} must be a list matching samples.")
        if field_name == "sample_node_ids":
            sample_node_ids = value
    if sample_node_ids is not None and any(node_id == "q" for node_id in sample_node_ids):
        raise ContractValidationError("Invalid training batch: sample_node_ids must not contain q.")


def validate_graph_rerank_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    if "type_weights" in config_dict:
        raise ContractValidationError(
            "Invalid graph rerank config: type_weights is deprecated; use neighbor_type_weights instead."
        )
    lambda_fields = ["lambda_init", "lambda_query", "lambda_neighbor", "lambda_bridge", "lambda_path"]
    for field_name in lambda_fields:
        value = config_dict.get(field_name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ContractValidationError(f"Invalid graph rerank config: {field_name} must be a finite non-negative number.")
    if float(config_dict.get("lambda_path", 0.0)) != 0.0:
        raise ContractValidationError("Invalid graph rerank config: lambda_path must remain 0.0 for HotpotQA Phase 1.")
    for field_name in ["seed_top_s", "max_hops"]:
        value = config_dict.get(field_name)
        if not isinstance(value, int) or value <= 0:
            raise ContractValidationError(f"Invalid graph rerank config: {field_name} must be a positive integer.")
    neighbor_type_weights = config_dict.get("neighbor_type_weights")
    if not isinstance(neighbor_type_weights, dict):
        raise ContractValidationError("Invalid graph rerank config: neighbor_type_weights must be an object.")
    unknown_neighbor_types = sorted(set(neighbor_type_weights) - NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES)
    if unknown_neighbor_types:
        raise ContractValidationError(
            f"Invalid graph rerank config: unsupported neighbor_type_weights entries={unknown_neighbor_types}."
        )
    for edge_type in sorted(NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES):
        if edge_type not in neighbor_type_weights:
            raise ContractValidationError(
                f"Invalid graph rerank config: missing neighbor type weight for edge_type={edge_type}."
            )
        value = neighbor_type_weights[edge_type]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise ContractValidationError(
                f"Invalid graph rerank config: neighbor_type_weights[{edge_type}] must be a finite non-negative number."
            )


def _validate_node_feature_config(value: object) -> None:
    feature_config = _to_plain_dict(value)
    _reject_unknown_fields(feature_config, NODE_FEATURE_CONFIG_FIELDS, "node feature config")
    for field_name in ["node_feature_names", "scorer_feature_names"]:
        feature_names = _validate_string_sequence(feature_config.get(field_name), field_name, allow_empty=True)
        unknown = sorted(set(feature_names) - KNOWN_NODE_FEATURES)
        if unknown:
            raise ContractValidationError(f"Invalid node feature config: unsupported {field_name}={unknown}.")


__all__ = [
    "validate_graph_batch",
    "validate_graph_rerank_config",
    "validate_rgcn_checkpoint_metadata",
    "validate_rgcn_model_config",
    "validate_rgcn_training_config",
    "validate_training_batch",
]
