from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_memory.config import CONFIG_LOADER
from graph_memory.registry import Registry
from graph_memory.retrieval_registry import get_method_spec, get_methods_requiring_dense_encoder
from graph_memory.training_config import device_from_training_config
from scripts.workflow.registry import is_dense_finetune_method


def attach_stage_config_projections(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["stage_configs"] = _build_stage_config_projections(manifest)
    return manifest


def _build_stage_config_projections(manifest: dict[str, Any]) -> dict[str, Any]:
    methods = list(manifest["selected_methods"])
    learned = manifest["artifacts"].get("learned", {})
    return {
        "pairs": {
            method: _stage_config_projection(Registry.configs.PAIRS, _pair_stage_argv(manifest, method))
            for method in methods
            if method in learned
        },
        "train": {
            method: _stage_config_projection(Registry.configs.TRAIN, _train_stage_argv(manifest, method))
            for method in methods
            if method in learned
        },
        "retrieve": {
            method: _stage_config_projection(Registry.configs.RETRIEVE, _retrieve_stage_argv(manifest, method))
            for method in methods
        },
        "evaluate": {
            method: _stage_config_projection(Registry.configs.EVALUATE, _evaluate_stage_argv(manifest, method))
            for method in methods
        },
    }


def _stage_config_projection(spec: Any, argv: Sequence[str]) -> dict[str, Any]:
    value = CONFIG_LOADER.to_json(CONFIG_LOADER.load(spec, argv))
    if not isinstance(value, dict):
        raise ValueError(f"Stage config projection must be an object: {spec.stage.value}")
    return _normalize_path_strings(value)


def _pair_stage_argv(manifest: dict[str, Any], method: str) -> list[str]:
    learned = manifest["artifacts"]["learned"][method]
    return [
        "--tasks",
        manifest["artifacts"]["inputs"]["train"]["input"],
        "--labels",
        manifest["artifacts"]["inputs"]["train"]["labels"],
        "--graphs",
        manifest["artifacts"]["graphs"]["train"],
        "--output",
        learned["train_pairs"],
        "--config",
        learned["effective_training_config"],
    ]


def _train_stage_argv(manifest: dict[str, Any], method: str) -> list[str]:
    learned = manifest["artifacts"]["learned"][method]
    argv = [
        "--method",
        method,
        "--train_tasks",
        manifest["artifacts"]["inputs"]["train"]["input"],
        "--train_labels",
        manifest["artifacts"]["inputs"]["train"]["labels"],
        "--train_pairs",
        learned["train_pairs"],
        "--dev_tasks",
        manifest["artifacts"]["inputs"]["dev"]["input"],
        "--dev_labels",
        manifest["artifacts"]["inputs"]["dev"]["labels"],
        "--output_dir",
        learned["training_output_dir"],
        "--config",
        learned["effective_training_config"],
    ]
    if is_dense_finetune_method(method):
        argv.extend(["--model_dir", learned["best_checkpoint"]])
    else:
        argv.extend(
            [
                "--train_graphs",
                manifest["artifacts"]["graphs"]["train"],
                "--dev_graphs",
                manifest["artifacts"]["graphs"]["dev"],
            ]
        )
    return argv


def _retrieve_stage_argv(manifest: dict[str, Any], method: str) -> list[str]:
    spec = get_method_spec(method)
    argv = [
        "--method",
        method,
        "--tasks",
        manifest["artifacts"]["inputs"]["test"]["input"],
        "--output",
        manifest["artifacts"]["predictions"][method],
        "--top_k",
        str(manifest["effective_config"]["top_k"]),
    ]
    if spec.requires_graphs:
        argv.extend(["--graphs", manifest["artifacts"]["graphs"]["test"]])
    if spec.requires_graph_config:
        argv.extend(["--graph_config", manifest["artifacts"]["tuned"][method]])
    if spec.requires_checkpoint:
        learned = manifest["artifacts"]["learned"][method]
        argv.extend(
            [
                "--checkpoint",
                learned["best_checkpoint"],
                "--device",
                device_from_training_config(manifest["effective_config"]["training"][method]),
            ]
        )
    if method in get_methods_requiring_dense_encoder() and not spec.requires_checkpoint:
        _append_dense_projection_args(argv, manifest)
    return argv


def _evaluate_stage_argv(manifest: dict[str, Any], method: str) -> list[str]:
    return [
        "--pred",
        manifest["artifacts"]["predictions"][method],
        "--labels",
        manifest["artifacts"]["inputs"]["test"]["labels"],
        "--graphs",
        manifest["artifacts"]["graphs"]["test"],
        "--output",
        manifest["artifacts"]["metrics"][method],
        "--failure_cases_output",
        manifest["artifacts"]["failure_cases"][method],
        "--failure_case_limit",
        "50",
    ]


def _append_dense_projection_args(argv: list[str], manifest: dict[str, Any]) -> None:
    config = manifest["effective_config"]
    argv.extend(
        [
            "--encoder_model",
            str(config["dense_encoder"]),
            "--query_prefix",
            str(config["query_prefix"]),
            "--passage_prefix",
            str(config["passage_prefix"]),
        ]
    )


def _normalize_path_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_path_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_path_strings(item) for item in value]
    if isinstance(value, str) and ("\\" in value or "/" in value):
        return Path(value).as_posix()
    return value


__all__ = ["attach_stage_config_projections"]
