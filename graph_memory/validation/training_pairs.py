from __future__ import annotations

from graph_memory.contracts.common import NEGATIVE_TRAIN_PAIR_SAMPLE_TYPES, TRAIN_PAIR_SAMPLE_TYPES
from graph_memory.validation.common import (
    ContractValidationError,
    _graph_node_ids,
    _memory_node_ids,
    _reject_unknown_fields,
    _require_record,
    _require_record_list,
    _require_record_map,
    _required_finite_number,
    _required_int,
    _required_string,
    _to_plain_dict,
)

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


__all__ = ["validate_negative_sampling_config", "validate_train_pair_build_summary", "validate_train_pairs"]
