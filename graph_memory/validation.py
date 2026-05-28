from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, TypeAlias, cast

from graph_memory.retrieval_registry import get_supported_methods
from graph_memory.types import (
    ALLOWED_EDGE_TYPES,
    ALLOWED_NODE_TYPES,
    NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES,
    NEIGHBOR_TYPE_WEIGHT_EDGE_TYPES,
    TRAIN_PAIR_SAMPLE_TYPES,
)


class ContractValidationError(ValueError):
    """Raised when an artifact violates a documented project contract."""


ValidationRecord: TypeAlias = dict[str, Any]
ValidationRecords: TypeAlias = list[ValidationRecord]
ValidationRecordMap: TypeAlias = dict[str, ValidationRecord]


FORBIDDEN_LABEL_FIELDS: set[str] = {
    "gold_answer",
    "gold_evidence_nodes",
    "gold_dependency_edges",
    "supporting_facts",
    "is_gold",
    "is_gold_evidence",
    "is_gold_edge",
}

MEMORY_TASK_INPUT_FIELDS = {"task_id", "query", "memory_items", "metadata", "debug"}
MEMORY_ITEM_FIELDS = {"id", "node_type", "text", "source", "sentence_id", "position"}
LABEL_FIELDS = {"task_id", "gold_answer", "gold_evidence_nodes", "gold_dependency_edges", "metadata", "debug"}
GRAPH_FIELDS = {"task_id", "nodes", "edges", "metadata", "debug"}
QUESTION_NODE_FIELDS = {"id", "node_type", "text"}
GRAPH_EDGE_FIELDS = {"source", "target", "edge_type", "weight", "directed"}
RANKED_RESULT_FIELDS = {
    "task_id",
    "method",
    "ranked_nodes",
    "retrieved_subgraph",
    "latency_ms",
    "input_tokens",
    "metadata",
    "debug",
}
RANKED_NODE_FIELDS = {"node_id", "score"}
RETRIEVED_SUBGRAPH_FIELDS = {"nodes", "edges"}
TRAIN_PAIR_FIELDS = {"task_id", "node_id", "label", "sample_type"}
TRAIN_PAIR_BUILD_SUMMARY_FIELDS = {
    "positive_count",
    "negative_count_by_type",
    "avg_positive_per_task",
    "avg_negative_per_task",
    "tasks_with_no_positive",
    "sampling_config",
}
NEGATIVE_SAMPLING_CONFIG_FIELDS = {
    "random_seed",
    "easy_random_per_positive",
    "hard_bm25_per_positive",
    "hard_dense_per_positive",
    "hard_graph_neighbor_per_positive",
    "hard_pool_size",
}
NODE_FEATURE_CONFIG_FIELDS = {"node_feature_names", "scorer_feature_names"}
TRAINABLE_MODEL_CONFIG_FIELDS = {
    "method_name",
    "encoder_model",
    "encoder_dim",
    "query_prefix",
    "passage_prefix",
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
TRAINABLE_TRAINING_CONFIG_FIELDS = {
    "optimizer_name",
    "learning_rate",
    "batch_size",
    "max_grad_norm",
    "random_seed",
    "pos_weight_enabled",
    "epochs",
}
TRAINABLE_CHECKPOINT_FIELDS = {
    "checkpoint_version",
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

METRIC_COLUMNS = [
    "Method",
    "Recall@2",
    "Recall@5",
    "Recall@10",
    "Evidence F1@5",
    "Evidence F1@10",
    "Full Support@5",
    "Full Support@10",
    "MRR",
    "Connected Evidence Recall@5",
    "Connected Evidence Recall@10",
    "Query-Evidence Connectivity@10",
    "Path Recall@10",
    "Edge Recall@10",
    "Retrieval Latency / Query",
]

def validate_no_label_fields(value: Any, *, artifact_name: str = "artifact", task_id: str | None = None) -> None:
    _walk_forbidden_fields(value, artifact_name=artifact_name, task_id=task_id, path=artifact_name)


def validate_memory_task_inputs(records: object) -> None:
    records = _require_record_list(records, "memory task inputs")

    seen_task_ids: set[str] = set()
    for index, task_input in enumerate(records):
        if not isinstance(task_input, dict):
            raise ContractValidationError(f"Invalid memory task inputs: record index={index} is not an object.")
        task_id = _required_string(task_input, "task_id", "memory task input")
        _reject_unknown_fields(task_input, MEMORY_TASK_INPUT_FIELDS, "memory task input", task_id)
        validate_no_label_fields(task_input, artifact_name="memory task input", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "memory task input task_id")
        _required_string(task_input, "query", "memory task input", task_id)

        memory_items = task_input.get("memory_items")
        if not isinstance(memory_items, list) or not memory_items:
            raise ContractValidationError(f"Invalid memory task input: task_id={task_id} memory_items must be non-empty.")

        seen_node_ids: set[str] = set()
        for expected_position, memory_item in enumerate(memory_items):
            if not isinstance(memory_item, dict):
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} memory item index={expected_position} is not an object."
                )
            _reject_unknown_fields(memory_item, MEMORY_ITEM_FIELDS, "memory item", task_id)
            node_id = _required_string(memory_item, "id", "memory item", task_id)
            _require_unique(node_id, seen_node_ids, f"memory item id task_id={task_id}")
            if node_id != f"m{expected_position}":
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} expected id=m{expected_position}."
                )
            if memory_item.get("node_type") != "document_sentence":
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} node_type must be document_sentence."
                )
            _required_string(memory_item, "text", "memory item", task_id)
            _required_string(memory_item, "source", "memory item", task_id)
            _required_int(memory_item, "sentence_id", "memory item", task_id, minimum=0)
            position = _required_int(memory_item, "position", "memory item", task_id, minimum=0)
            if position != expected_position:
                raise ContractValidationError(
                    f"Invalid memory task input: task_id={task_id} node_id={node_id} position={position} expected {expected_position}."
                )


def validate_memory_task_labels(records: object, inputs_by_task_id: object) -> None:
    records = _require_record_list(records, "memory task labels")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")

    seen_task_ids: set[str] = set()
    for index, task_labels in enumerate(records):
        if not isinstance(task_labels, dict):
            raise ContractValidationError(f"Invalid memory task labels: record index={index} is not an object.")
        task_id = _required_string(task_labels, "task_id", "memory task labels")
        _reject_unknown_fields(task_labels, LABEL_FIELDS, "memory task labels", task_id)
        _require_unique(task_id, seen_task_ids, "memory task labels task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid memory task labels: task_id={task_id} has no matching input task.")

        _required_string(task_labels, "gold_answer", "memory task labels", task_id)
        gold_nodes = task_labels.get("gold_evidence_nodes")
        if not isinstance(gold_nodes, list) or not gold_nodes:
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_evidence_nodes must be a non-empty list."
            )
        if len(gold_nodes) != len(set(gold_nodes)):
            raise ContractValidationError(f"Invalid memory task labels: task_id={task_id} duplicate gold evidence node.")

        valid_node_ids = _memory_node_ids(inputs_by_task_id[task_id])
        for node_id in gold_nodes:
            if not isinstance(node_id, str) or node_id not in valid_node_ids:
                raise ContractValidationError(
                    f"Invalid memory task labels: task_id={task_id} gold node={node_id} does not exist in input task."
                )

        dependency_edges = task_labels.get("gold_dependency_edges")
        if not isinstance(dependency_edges, list):
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_dependency_edges must be a list."
            )
        if dependency_edges:
            raise ContractValidationError(
                f"Invalid memory task labels: task_id={task_id} gold_dependency_edges must be empty for HotpotQA Phase 1."
            )


def validate_graphs(graphs: object, inputs_by_task_id: object) -> None:
    graphs = _require_record_list(graphs, "graphs")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")

    seen_task_ids: set[str] = set()
    for index, graph in enumerate(graphs):
        if not isinstance(graph, dict):
            raise ContractValidationError(f"Invalid graph: record index={index} is not an object.")
        task_id = _required_string(graph, "task_id", "graph")
        _reject_unknown_fields(graph, GRAPH_FIELDS, "graph", task_id)
        validate_no_label_fields(graph, artifact_name="graph", task_id=task_id)
        _require_unique(task_id, seen_task_ids, "graph task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid graph: task_id={task_id} has no matching input task.")

        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} nodes must be a list.")
        if not isinstance(edges, list):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} edges must be a list.")

        graph_node_ids = _validate_graph_nodes(nodes, inputs_by_task_id[task_id], task_id)
        for edge in edges:
            _validate_graph_edge(edge, graph_node_ids, task_id)


def validate_ranked_results(predictions: object, inputs_by_task_id: object) -> None:
    predictions = _require_record_list(predictions, "ranked results")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")

    seen_task_ids: set[str] = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ContractValidationError(f"Invalid ranked results: record index={index} is not an object.")
        task_id = _required_string(prediction, "task_id", "ranked result")
        _reject_unknown_fields(prediction, RANKED_RESULT_FIELDS, "ranked result", task_id)
        _require_unique(task_id, seen_task_ids, "ranked result task_id")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} has no matching input task.")

        method = _required_string(prediction, "method", "ranked result", task_id)
        if method not in _supported_methods():
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} unsupported method={method}.")

        ranked_nodes = prediction.get("ranked_nodes")
        if not isinstance(ranked_nodes, list):
            raise ContractValidationError(f"Invalid ranked results: task_id={task_id} ranked_nodes must be a list.")
        expected_node_ids = _memory_node_ids(inputs_by_task_id[task_id])
        seen_node_ids: set[str] = set()
        previous_score: float | None = None
        for ranked_node in ranked_nodes:
            if not isinstance(ranked_node, dict):
                raise ContractValidationError(f"Invalid ranked results: task_id={task_id} ranked node is not an object.")
            _reject_unknown_fields(ranked_node, RANKED_NODE_FIELDS, "ranked node", task_id)
            node_id = _required_string(ranked_node, "node_id", "ranked node", task_id)
            if node_id in seen_node_ids:
                raise ContractValidationError(
                    f"Invalid ranked results: task_id={task_id} method={method} ranked_nodes contains duplicate node_id={node_id}."
                )
            seen_node_ids.add(node_id)
            if node_id not in expected_node_ids:
                raise ContractValidationError(
                    f"Invalid ranked results: task_id={task_id} method={method} ranked node_id={node_id} does not exist."
                )
            score = _required_finite_number(ranked_node, "score", "ranked node", task_id)
            if previous_score is not None and score > previous_score:
                raise ContractValidationError(
                    f"Invalid ranked results: task_id={task_id} method={method} ranked_nodes must be sorted descending."
                )
            previous_score = score

        if seen_node_ids != expected_node_ids:
            missing = sorted(expected_node_ids - seen_node_ids)
            extra = sorted(seen_node_ids - expected_node_ids)
            raise ContractValidationError(
                f"Invalid ranked results: task_id={task_id} method={method} ranking must include every memory node exactly once; missing={missing} extra={extra}."
            )

        _required_finite_number(prediction, "latency_ms", "ranked result", task_id, minimum=0.0)
        _required_int(prediction, "input_tokens", "ranked result", task_id, minimum=0)
        _validate_retrieved_subgraph(prediction.get("retrieved_subgraph"), expected_node_ids, task_id)


def validate_train_pairs(
    records: object,
    inputs_by_task_id: object,
    labels_by_task_id: object,
    graphs_by_task_id: object,
) -> None:
    records = _require_record_list(records, "train pairs")
    inputs_by_task_id = _require_record_map(inputs_by_task_id, "memory task inputs by task_id")
    labels_by_task_id = _require_record_map(labels_by_task_id, "memory task labels by task_id")
    graphs_by_task_id = _require_record_map(graphs_by_task_id, "graphs by task_id")

    seen_keys: set[tuple[str, str, str]] = set()
    positive_nodes_by_task_id: dict[str, set[str]] = {task_id: set() for task_id in labels_by_task_id}
    for index, pair in enumerate(records):
        if not isinstance(pair, dict):
            raise ContractValidationError(f"Invalid train pairs: record index={index} is not an object.")
        _reject_unknown_fields(pair, TRAIN_PAIR_FIELDS, "train pair")
        task_id = _required_string(pair, "task_id", "train pair")
        if task_id not in inputs_by_task_id:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} has no matching input task.")
        if task_id not in labels_by_task_id:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} has no matching label task.")
        if task_id not in graphs_by_task_id:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} has no matching graph.")

        node_id = _required_string(pair, "node_id", "train pair", task_id)
        if node_id == "q":
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} node_id=q is not a memory node.")
        memory_node_ids = _memory_node_ids(inputs_by_task_id[task_id])
        if node_id not in memory_node_ids:
            raise ContractValidationError(
                f"Invalid train pairs: task_id={task_id} node_id={node_id} does not exist in input task."
            )
        if node_id not in _graph_node_ids(graphs_by_task_id[task_id]) - {"q"}:
            raise ContractValidationError(
                f"Invalid train pairs: task_id={task_id} node_id={node_id} does not exist in graph."
            )

        label = _required_int(pair, "label", "train pair", task_id, minimum=0)
        if label not in {0, 1}:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} label must be 0 or 1.")
        sample_type = _required_string(pair, "sample_type", "train pair", task_id)
        if sample_type not in TRAIN_PAIR_SAMPLE_TYPES:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} unsupported sample_type={sample_type}.")

        duplicate_key = (task_id, node_id, sample_type)
        if duplicate_key in seen_keys:
            raise ContractValidationError(
                f"Invalid train pairs: duplicate (task_id, node_id, sample_type)={duplicate_key}."
            )
        seen_keys.add(duplicate_key)

        gold_nodes = set(labels_by_task_id[task_id].get("gold_evidence_nodes", []))
        if sample_type == "positive" and label != 1:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} positive sample requires label=1.")
        if sample_type != "positive" and label != 0:
            raise ContractValidationError(f"Invalid train pairs: task_id={task_id} negative sample requires label=0.")
        if label == 1:
            if node_id not in gold_nodes:
                raise ContractValidationError(
                    f"Invalid train pairs: task_id={task_id} positive node_id={node_id} is not gold evidence."
                )
            positive_nodes_by_task_id.setdefault(task_id, set()).add(node_id)
        if label == 0 and node_id in gold_nodes:
            raise ContractValidationError(
                f"Invalid train pairs: task_id={task_id} negative node_id={node_id} is gold evidence."
            )

    for task_id, labels in labels_by_task_id.items():
        gold_nodes = set(labels.get("gold_evidence_nodes", []))
        observed_positive_nodes = positive_nodes_by_task_id.get(task_id, set())
        if observed_positive_nodes != gold_nodes:
            missing = sorted(gold_nodes - observed_positive_nodes)
            extra = sorted(observed_positive_nodes - gold_nodes)
            raise ContractValidationError(
                f"Invalid train pairs: task_id={task_id} positives must exactly match gold evidence; missing={missing} extra={extra}."
            )


def validate_negative_sampling_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    _reject_unknown_fields(config_dict, NEGATIVE_SAMPLING_CONFIG_FIELDS, "negative sampling config")
    _required_int(config_dict, "random_seed", "negative sampling config")
    hard_count = 0
    for field_name in [
        "easy_random_per_positive",
        "hard_bm25_per_positive",
        "hard_dense_per_positive",
        "hard_graph_neighbor_per_positive",
    ]:
        value = _required_int(config_dict, field_name, "negative sampling config", minimum=0)
        if field_name.startswith("hard_"):
            hard_count += value
    hard_pool_size = _required_int(config_dict, "hard_pool_size", "negative sampling config", minimum=1)
    if hard_count > 0 and hard_pool_size <= 0:
        raise ContractValidationError("Invalid negative sampling config: hard_pool_size must be positive.")


def validate_train_pair_build_summary(summary: object) -> None:
    summary = _require_record(summary, "train pair build summary")
    _reject_unknown_fields(summary, TRAIN_PAIR_BUILD_SUMMARY_FIELDS, "train pair build summary")
    _required_int(summary, "positive_count", "train pair build summary", minimum=0)

    negative_count_by_type = summary.get("negative_count_by_type")
    if not isinstance(negative_count_by_type, dict):
        raise ContractValidationError("Invalid train pair build summary: negative_count_by_type must be an object.")
    unknown_sample_types = sorted(set(negative_count_by_type) - NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES)
    if unknown_sample_types:
        raise ContractValidationError(
            f"Invalid train pair build summary: unsupported negative sample types={unknown_sample_types}."
        )
    for sample_type, count in negative_count_by_type.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ContractValidationError(
                f"Invalid train pair build summary: negative_count_by_type[{sample_type}] must be a non-negative integer."
            )

    for field_name in ["avg_positive_per_task", "avg_negative_per_task"]:
        _required_finite_number(summary, field_name, "train pair build summary", minimum=0.0)
    tasks_with_no_positive = summary.get("tasks_with_no_positive")
    if not isinstance(tasks_with_no_positive, list):
        raise ContractValidationError("Invalid train pair build summary: tasks_with_no_positive must be a list.")
    for task_id in tasks_with_no_positive:
        if not isinstance(task_id, str) or not task_id:
            raise ContractValidationError("Invalid train pair build summary: tasks_with_no_positive entries must be strings.")
    if tasks_with_no_positive:
        raise ContractValidationError("Invalid train pair build summary: tasks_with_no_positive must be empty.")

    sampling_config = summary.get("sampling_config")
    if not isinstance(sampling_config, dict):
        raise ContractValidationError("Invalid train pair build summary: sampling_config must be an object.")
    validate_negative_sampling_config(sampling_config)


def validate_trainable_model_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    _reject_unknown_fields(config_dict, TRAINABLE_MODEL_CONFIG_FIELDS, "trainable model config")
    _required_string(config_dict, "method_name", "trainable model config")
    _required_string(config_dict, "encoder_model", "trainable model config")
    _required_int(config_dict, "encoder_dim", "trainable model config", minimum=1)
    _required_string(config_dict, "query_prefix", "trainable model config")
    _required_string(config_dict, "passage_prefix", "trainable model config")
    _required_int(config_dict, "hidden_dim", "trainable model config", minimum=1)
    _required_int(config_dict, "num_layers", "trainable model config", minimum=0)
    dropout = _required_finite_number(config_dict, "dropout", "trainable model config", minimum=0.0)
    if dropout >= 1.0:
        raise ContractValidationError("Invalid trainable model config: dropout must be < 1.0.")

    _validate_node_feature_config(config_dict.get("feature_config"))
    _validate_string_sequence(config_dict.get("relation_vocab"), "relation_vocab", allow_empty=False)
    graph_encoder_type = _required_string(config_dict, "graph_encoder_type", "trainable model config")
    if graph_encoder_type not in {"identity", "rgcn"}:
        raise ContractValidationError("Invalid trainable model config: graph_encoder_type must be identity or rgcn.")
    message_transform_type = _required_string(config_dict, "message_transform_type", "trainable model config")
    if message_transform_type not in {"typed", "shared"}:
        raise ContractValidationError("Invalid trainable model config: message_transform_type must be typed or shared.")
    edge_weight_policy = _required_string(config_dict, "edge_weight_policy", "trainable model config")
    if edge_weight_policy not in {"artifact", "uniform"}:
        raise ContractValidationError("Invalid trainable model config: edge_weight_policy must be artifact or uniform.")
    enabled_edge_types = set(_validate_string_sequence(config_dict.get("enabled_edge_types"), "enabled_edge_types", allow_empty=True))
    unknown_edge_types = sorted(enabled_edge_types - ALLOWED_EDGE_TYPES)
    if unknown_edge_types:
        raise ContractValidationError(
            f"Invalid trainable model config: unsupported enabled_edge_types={unknown_edge_types}."
        )
    _required_string(config_dict, "ablation_name", "trainable model config")


def validate_trainable_training_config(config: object) -> None:
    config_dict = _to_plain_dict(config)
    _reject_unknown_fields(config_dict, TRAINABLE_TRAINING_CONFIG_FIELDS, "trainable training config")
    if _required_string(config_dict, "optimizer_name", "trainable training config") != "AdamW":
        raise ContractValidationError("Invalid trainable training config: optimizer_name must be AdamW.")
    _required_finite_number(config_dict, "learning_rate", "trainable training config", minimum=0.0)
    _required_int(config_dict, "batch_size", "trainable training config", minimum=1)
    _required_finite_number(config_dict, "max_grad_norm", "trainable training config", minimum=0.0)
    _required_int(config_dict, "random_seed", "trainable training config")
    if not isinstance(config_dict.get("pos_weight_enabled"), bool):
        raise ContractValidationError("Invalid trainable training config: pos_weight_enabled must be boolean.")
    _required_int(config_dict, "epochs", "trainable training config", minimum=1)


def validate_trainable_checkpoint_metadata(checkpoint: object, *, expected_method: str | None = None) -> None:
    checkpoint = _require_record(checkpoint, "trainable checkpoint")
    _reject_unknown_fields(checkpoint, TRAINABLE_CHECKPOINT_FIELDS, "trainable checkpoint")
    version = _required_int(checkpoint, "checkpoint_version", "trainable checkpoint", minimum=1)
    if version != 1:
        raise ContractValidationError(f"Invalid trainable checkpoint: unsupported checkpoint_version={version}.")
    method_name = _required_string(checkpoint, "method_name", "trainable checkpoint")
    if expected_method is not None and method_name != expected_method:
        raise ContractValidationError(
            f"Invalid trainable checkpoint: method_name={method_name} does not match expected_method={expected_method}."
        )
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ContractValidationError("Invalid trainable checkpoint: model_state_dict must be present.")
    if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
        raise ContractValidationError("Invalid trainable checkpoint: optimizer_state_dict must be present.")
    if not isinstance(checkpoint.get("scheduler_state_dict"), dict):
        raise ContractValidationError("Invalid trainable checkpoint: scheduler_state_dict must be present.")
    _required_int(checkpoint, "epoch", "trainable checkpoint", minimum=0)
    _required_int(checkpoint, "global_step", "trainable checkpoint", minimum=0)
    _required_finite_number(checkpoint, "best_dev_metric", "trainable checkpoint")
    validate_trainable_model_config(checkpoint.get("model_config"))
    validate_trainable_training_config(checkpoint.get("training_config"))
    _required_string(checkpoint, "created_at", "trainable checkpoint")


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


def validate_metric_rows(rows: object) -> None:
    rows = _require_record_list(rows, "metric rows")
    for row in rows:
        if not isinstance(row, dict):
            raise ContractValidationError("Invalid metric rows: row is not an object.")
        missing = [column for column in METRIC_COLUMNS if column not in row]
        if missing:
            raise ContractValidationError(f"Invalid metric rows: missing columns={missing}.")
        for column in METRIC_COLUMNS:
            if column in {"Method", "Path Recall@10", "Edge Recall@10"}:
                continue
            value = float(row[column])
            if not math.isfinite(value):
                raise ContractValidationError(f"Invalid metric rows: column={column} must be finite.")
            if column == "Retrieval Latency / Query":
                if value < 0.0:
                    raise ContractValidationError("Invalid metric rows: latency must be non-negative.")
            elif value < 0.0 or value > 1.0:
                raise ContractValidationError(f"Invalid metric rows: column={column} must be in [0.0, 1.0].")


def validate_task_id_alignment(artifact_name: str, expected_task_ids: set[str], observed_task_ids: set[str]) -> None:
    if expected_task_ids != observed_task_ids:
        missing = sorted(expected_task_ids - observed_task_ids)
        extra = sorted(observed_task_ids - expected_task_ids)
        raise ContractValidationError(
            f"Invalid {artifact_name}: task_id alignment mismatch; missing={missing} extra={extra}."
        )


def _validate_graph_nodes(nodes: list[object], task_input: ValidationRecord, task_id: str) -> set[str]:
    seen_node_ids: set[str] = set()
    question_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            raise ContractValidationError(f"Invalid graph: task_id={task_id} node is not an object.")
        node_id = _required_string(node, "id", "graph node", task_id)
        _require_unique(node_id, seen_node_ids, f"graph node id task_id={task_id}")
        node_type = _required_string(node, "node_type", "graph node", task_id)
        if node_type not in ALLOWED_NODE_TYPES:
            raise ContractValidationError(f"Invalid graph: task_id={task_id} node_id={node_id} unsupported node_type={node_type}.")
        if node_id == "q":
            question_count += 1
            _reject_unknown_fields(node, QUESTION_NODE_FIELDS, "question graph node", task_id)
            if node_type != "question":
                raise ContractValidationError(f"Invalid graph: task_id={task_id} q node must have node_type=question.")
            _required_string(node, "text", "question graph node", task_id)
        else:
            _reject_unknown_fields(node, MEMORY_ITEM_FIELDS, "memory graph node", task_id)
            if node_type != "document_sentence":
                raise ContractValidationError(
                    f"Invalid graph: task_id={task_id} node_id={node_id} memory node must have node_type=document_sentence."
                )

    if question_count != 1:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} must contain exactly one q node.")

    expected_memory_node_ids = _memory_node_ids(task_input)
    observed_memory_node_ids = seen_node_ids - {"q"}
    if observed_memory_node_ids != expected_memory_node_ids:
        missing = sorted(expected_memory_node_ids - observed_memory_node_ids)
        extra = sorted(observed_memory_node_ids - expected_memory_node_ids)
        raise ContractValidationError(
            f"Invalid graph: task_id={task_id} graph memory nodes mismatch; missing={missing} extra={extra}."
        )
    return seen_node_ids


def _validate_graph_edge(edge: ValidationRecord, graph_node_ids: set[str], task_id: str) -> None:
    if not isinstance(edge, dict):
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge is not an object.")
    _reject_unknown_fields(edge, GRAPH_EDGE_FIELDS, "graph edge", task_id)
    source = _required_string(edge, "source", "graph edge", task_id)
    target = _required_string(edge, "target", "graph edge", task_id)
    if source not in graph_node_ids:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge source={source} does not exist in nodes.")
    if target not in graph_node_ids:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge target={target} does not exist in nodes.")
    edge_type = _required_string(edge, "edge_type", "graph edge", task_id)
    if edge_type not in ALLOWED_EDGE_TYPES:
        raise ContractValidationError(f"Invalid graph: task_id={task_id} unsupported edge_type={edge_type}.")
    _required_finite_number(edge, "weight", "graph edge", task_id, minimum=0.0)
    if not isinstance(edge.get("directed"), bool):
        raise ContractValidationError(f"Invalid graph: task_id={task_id} edge directed must be boolean.")


def _validate_retrieved_subgraph(value: Any, valid_node_ids: set[str], task_id: str) -> None:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} retrieved_subgraph must be an object.")
    _reject_unknown_fields(value, RETRIEVED_SUBGRAPH_FIELDS, "retrieved subgraph", task_id)
    nodes = value.get("nodes")
    edges = value.get("edges")
    if not isinstance(nodes, list):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} retrieved_subgraph.nodes must be a list.")
    if not isinstance(edges, list):
        raise ContractValidationError(f"Invalid ranked results: task_id={task_id} retrieved_subgraph.edges must be a list.")
    subgraph_node_ids = set(nodes)
    for node_id in subgraph_node_ids:
        if node_id not in valid_node_ids and node_id != "q":
            raise ContractValidationError(
                f"Invalid ranked results: task_id={task_id} retrieved_subgraph node_id={node_id} does not exist."
            )
    for edge in edges:
        edge_node_ids = subgraph_node_ids | {"q"}
        _validate_graph_edge(edge, edge_node_ids, task_id)


def _walk_forbidden_fields(value: Any, *, artifact_name: str, task_id: str | None, path: str) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in FORBIDDEN_LABEL_FIELDS:
                location = f" task_id={task_id}" if task_id is not None else ""
                raise ContractValidationError(
                    f"Invalid {artifact_name}:{location} forbidden label field {key} at {path}.{key}."
                )
            _walk_forbidden_fields(nested_value, artifact_name=artifact_name, task_id=task_id, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _walk_forbidden_fields(nested_value, artifact_name=artifact_name, task_id=task_id, path=f"{path}[{index}]")


def _require_record_list(value: object, artifact_name: str) -> ValidationRecords:
    if not isinstance(value, list):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be a list.")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ContractValidationError(f"Invalid {artifact_name}: record index={index} is not an object.")
    return cast(ValidationRecords, value)


def _require_record_map(value: object, artifact_name: str) -> ValidationRecordMap:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be an object.")
    for key, record in value.items():
        if not isinstance(key, str) or not key:
            raise ContractValidationError(f"Invalid {artifact_name}: keys must be non-empty strings.")
        if not isinstance(record, dict):
            raise ContractValidationError(f"Invalid {artifact_name}: value for key={key} is not an object.")
    return cast(ValidationRecordMap, value)


def _require_record(value: object, artifact_name: str) -> ValidationRecord:
    if not isinstance(value, dict):
        raise ContractValidationError(f"Invalid {artifact_name}: artifact must be an object.")
    return cast(ValidationRecord, value)


def _reject_unknown_fields(
    record: ValidationRecord,
    allowed_fields: set[str],
    artifact_name: str,
    task_id: str | None = None,
) -> None:
    unknown = sorted(set(record) - allowed_fields)
    if unknown:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} unknown fields={unknown}.")


def _required_string(record: ValidationRecord, field_name: str, artifact_name: str, task_id: str | None = None) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be a non-empty string.")
    return value


def _required_int(
    record: ValidationRecord,
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
    *,
    minimum: int | None = None,
) -> int:
    value = record.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be an integer.")
    if minimum is not None and value < minimum:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be >= {minimum}.")
    return value


def _required_finite_number(
    record: ValidationRecord,
    field_name: str,
    artifact_name: str,
    task_id: str | None = None,
    *,
    minimum: float | None = None,
) -> float:
    value = record.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be finite.")
    number = float(value)
    if minimum is not None and number < minimum:
        location = f" task_id={task_id}" if task_id is not None else ""
        raise ContractValidationError(f"Invalid {artifact_name}:{location} field={field_name} must be >= {minimum}.")
    return number


def _require_unique(value: str, seen_values: set[str], artifact_name: str) -> None:
    if value in seen_values:
        raise ContractValidationError(f"Invalid {artifact_name}: duplicate value={value}.")
    seen_values.add(value)


def _memory_node_ids(task_input: ValidationRecord) -> set[str]:
    memory_items = task_input.get("memory_items")
    if not isinstance(memory_items, list):
        return set()
    return {memory_item["id"] for memory_item in memory_items if isinstance(memory_item, dict) and "id" in memory_item}


def _graph_node_ids(graph: ValidationRecord) -> set[str]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return set()
    return {node["id"] for node in nodes if isinstance(node, dict) and "id" in node}


def _supported_methods() -> tuple[str, ...]:
    return get_supported_methods()


def _to_plain_dict(config: object) -> ValidationRecord:
    if isinstance(config, dict):
        return cast(ValidationRecord, config)
    if is_dataclass(config) and not isinstance(config, type):
        return cast(ValidationRecord, asdict(config))
    raise ContractValidationError("Invalid config: expected dict or dataclass instance.")


def _validate_node_feature_config(value: object) -> None:
    feature_config = _to_plain_dict(value)
    _reject_unknown_fields(feature_config, NODE_FEATURE_CONFIG_FIELDS, "node feature config")
    for field_name in ["node_feature_names", "scorer_feature_names"]:
        feature_names = _validate_string_sequence(feature_config.get(field_name), field_name, allow_empty=True)
        unknown = sorted(set(feature_names) - KNOWN_NODE_FEATURES)
        if unknown:
            raise ContractValidationError(f"Invalid node feature config: unsupported {field_name}={unknown}.")


def _validate_string_sequence(value: object, field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"Invalid trainable model config: {field_name} must be a list or tuple.")
    if not allow_empty and not value:
        raise ContractValidationError(f"Invalid trainable model config: {field_name} must be non-empty.")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ContractValidationError(f"Invalid trainable model config: {field_name} entries must be non-empty strings.")
        strings.append(item)
    if len(strings) != len(set(strings)):
        raise ContractValidationError(f"Invalid trainable model config: {field_name} contains duplicate entries.")
    return tuple(strings)


def _required_attr(value: object, field_name: str, artifact_name: str) -> Any:
    if not hasattr(value, field_name):
        raise ContractValidationError(f"Invalid {artifact_name}: missing field={field_name}.")
    return getattr(value, field_name)


def _require_tensor_1d(value: object, field_name: str, artifact_name: str) -> Any:
    tensor = _required_attr(value, field_name, artifact_name)
    if getattr(tensor, "ndim", None) != 1:
        raise ContractValidationError(f"Invalid {artifact_name}: {field_name} must be a 1D tensor.")
    return tensor


def _require_tensor_2d(value: object, field_name: str, artifact_name: str) -> Any:
    tensor = _required_attr(value, field_name, artifact_name)
    if getattr(tensor, "ndim", None) != 2:
        raise ContractValidationError(f"Invalid {artifact_name}: {field_name} must be a 2D tensor.")
    return tensor
