from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_memory.io import merge_config, read_json, write_json
from graph_memory.observability import now_iso
from graph_memory.retrieval_registry import (
    get_graph_rerank_methods,
    get_method_spec,
    get_methods_requiring_dense_encoder,
    get_supported_methods,
)
from graph_memory.training_config import load_trainable_training_config, device_from_training_config

STAGE_ORDER = ("prepare", "graphs", "pairs", "tune", "train", "retrieve", "evaluate", "aggregate")
STAGE_DESCRIPTIONS = {
    "prepare": "Build split-specific task, label, and combined input artifacts.",
    "graphs": "Build evidence graph artifacts for train, dev, and test splits.",
    "pairs": "Build supervised training pairs for checkpoint-backed methods.",
    "tune": "Select graph-rerank parameters from the search-space config.",
    "train": "Train checkpoint-backed graph retrievers.",
    "retrieve": "Write ranked evidence predictions for selected methods.",
    "evaluate": "Evaluate ranked predictions and write per-method metrics.",
    "aggregate": "Aggregate per-method metrics into report tables.",
}
CONFIG_ROOT = Path("configs")
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
SEARCH_SPACE_CONFIG_DIR = CONFIG_ROOT / "search_spaces"
TRAINING_CONFIG_DIR = CONFIG_ROOT / "training"
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/hotpotqa_evidence_retrieval.json")
DEFAULT_SEARCH_SPACE_CONFIG = Path("configs/search_spaces/graph_rerank.json")


@dataclass(frozen=True)
class StageCommand:
    stage: str
    argv: list[str]
    method: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class ConfigEntry:
    kind: str
    name: str
    path: str


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_experiment_config_path(path)
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {config_path}")
    return config


def resolve_experiment_config_path(path: str | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_EXPERIMENT_CONFIG
    value = str(path)
    candidate = Path(value)
    if candidate.exists() or _looks_like_explicit_path(value):
        return candidate
    config_name = candidate.stem if candidate.suffix == ".json" else value
    return EXPERIMENT_CONFIG_DIR / f"{config_name}.json"


def resolve_training_config_path(method: str, path: str | Path) -> Path:
    value = str(path)
    candidate = Path(value)
    if candidate.exists():
        return candidate
    normalized = value.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) == 1 and candidate.suffix == ".json":
        return TRAINING_CONFIG_DIR / method / candidate.name
    if len(parts) == 2 and parts[0] in get_supported_methods():
        method_name, config_name = parts
        name = Path(config_name).stem if Path(config_name).suffix == ".json" else config_name
        return TRAINING_CONFIG_DIR / method_name / f"{name}.json"
    if _looks_like_explicit_training_path(value):
        return candidate
    config_name = candidate.stem if candidate.suffix == ".json" else value
    return TRAINING_CONFIG_DIR / method / f"{config_name}.json"


def initialize_experiment(
    experiment_name: str,
    *,
    config: dict[str, Any],
    run_root: str | Path = "runs",
    profile: str | None = None,
    methods: Sequence[str] | None = None,
    stages: Sequence[str] | None = None,
    cli_overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_root) / experiment_name
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        if not isinstance(existing, dict):
            raise ValueError(f"Existing manifest must be a JSON object: {manifest_path}")
        _reject_config_change(existing, config=config, profile=profile, methods=methods, cli_overrides=cli_overrides)
        return existing

    effective_config = build_effective_config(config, profile=profile, cli_overrides=cli_overrides)
    selected_methods = list(methods) if methods is not None else list(effective_config["methods"])
    _validate_methods(selected_methods)
    _attach_resolved_training_configs(effective_config, selected_methods)
    selected_stages = _select_stages(stages, from_stage=None, to_stage=None, methods=selected_methods)
    artifacts = _build_artifact_paths(run_dir, selected_methods)
    manifest = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "recipe": effective_config["recipe"],
        "profile": profile or effective_config.get("profile"),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "paths": {
            "run_dir": _path_str(run_dir),
            "manifest": _path_str(manifest_path),
            "effective_config": _path_str(run_dir / "config" / "effective_config.json"),
        },
        "selected_methods": selected_methods,
        "selected_stages": selected_stages,
        "effective_config": effective_config,
        "artifacts": artifacts,
        "stage_status": {},
    }
    _write_resolved_training_configs(manifest)
    write_json(run_dir / "config" / "effective_config.json", effective_config)
    write_json(manifest_path, manifest)
    return manifest


def load_manifest(experiment_name: str, *, run_root: str | Path = "runs") -> dict[str, Any]:
    manifest_path = Path(run_root) / experiment_name / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    return manifest


def build_effective_config(
    config: dict[str, Any],
    *,
    profile: str | None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_name = profile or str(config.get("default_profile", "quick"))
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}")

    defaults = dict(config.get("defaults", {}))
    profile_config = dict(profiles[profile_name])
    split_offsets = config.get("split_offsets", {})
    split_sources = config.get("split_sources", {"train": "train", "dev": "dev", "test": "dev"})
    base_config = {
        "recipe": config.get("recipe", "hotpotqa_evidence_retrieval"),
        "dataset": config.get("dataset", "hotpotqa"),
        "task": config.get("task", "evidence_retrieval"),
        "profile": profile_name,
        "raw": config["raw"],
        "graph": config.get("graph", {}),
        "methods": config.get("methods", list(get_supported_methods())),
        "search_spaces": config.get("search_spaces", {"graph_rerank": str(DEFAULT_SEARCH_SPACE_CONFIG)}),
        "training_configs": config.get("training_configs", {}),
        **defaults,
        "splits": {
            "train": {
                "source": split_sources.get("train", "train"),
                "max_examples": profile_config["train_examples"],
                "seed": defaults.get("seed", 13),
                "offset": split_offsets.get("train", 0),
            },
            "dev": {
                "source": split_sources.get("dev", "dev"),
                "max_examples": profile_config["dev_examples"],
                "seed": defaults.get("seed", 13),
                "offset": split_offsets.get("dev", 0),
            },
            "test": {
                "source": split_sources.get("test", "dev"),
                "max_examples": profile_config["test_examples"],
                "seed": defaults.get("seed", 13),
                "offset": split_offsets.get("test", 500),
            },
        },
    }
    return merge_config(base_config, None, cli_overrides)


def _attach_resolved_training_configs(effective_config: dict[str, Any], selected_methods: Sequence[str]) -> None:
    training_config_paths = effective_config.get("training_configs", {})
    if not isinstance(training_config_paths, dict):
        raise ValueError("Experiment config training_configs must be an object.")

    resolved_by_method: dict[str, Any] = {}
    for method in selected_methods:
        spec = get_method_spec(method)
        if not spec.requires_checkpoint:
            continue
        config_path = training_config_paths.get(method)
        if not isinstance(config_path, str) or not config_path:
            raise ValueError(f"Trainable method={method} requires a training config path.")
        resolved_config_path = resolve_training_config_path(method, config_path)
        resolved = load_trainable_training_config(resolved_config_path, profile=str(effective_config["profile"]))
        if resolved["method"] != method:
            raise ValueError(
                f"Training config method={resolved['method']} does not match selected method={method}."
            )
        resolved_by_method[method] = resolved

    if resolved_by_method:
        effective_config["training"] = resolved_by_method


def _write_resolved_training_configs(manifest: dict[str, Any]) -> None:
    training_configs = manifest["effective_config"].get("training", {})
    if not isinstance(training_configs, dict):
        return
    for method, training_config in training_configs.items():
        learned_artifacts = manifest["artifacts"]["learned"][method]
        write_json(learned_artifacts["effective_training_config"], training_config)


def build_stage_plan(
    manifest: dict[str, Any],
    *,
    stages: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[StageCommand]:
    selected_methods = list(methods) if methods is not None else list(manifest["selected_methods"])
    _validate_methods(selected_methods)
    selected_stages = _select_stages(stages, from_stage=from_stage, to_stage=to_stage, methods=selected_methods)
    _validate_stage_dependencies(manifest, selected_stages, selected_methods)

    commands: list[StageCommand] = []
    if "prepare" in selected_stages:
        commands.extend(_prepare_commands(manifest))
    if "graphs" in selected_stages:
        commands.extend(_graph_commands(manifest))
    if "pairs" in selected_stages:
        commands.extend(_pair_commands(manifest, selected_methods))
    if "tune" in selected_stages:
        commands.extend(_tune_commands(manifest, selected_methods))
    if "train" in selected_stages:
        commands.extend(_train_commands(manifest, selected_methods))
    if "retrieve" in selected_stages:
        commands.extend(_retrieve_commands(manifest, selected_methods))
    if "evaluate" in selected_stages:
        commands.extend(_evaluate_commands(manifest, selected_methods))
    if "aggregate" in selected_stages:
        commands.append(_aggregate_command(manifest))
    return commands


def run_stage_plan(commands: Sequence[StageCommand]) -> None:
    for command in commands:
        subprocess.run(command.argv, check=True)


def update_manifest_status(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = inspect_experiment_status(manifest)
    manifest["stage_status"] = {_status_key(row): row for row in rows}
    manifest["updated_at"] = now_iso()
    write_json(manifest["paths"]["manifest"], manifest)
    return manifest


def inspect_experiment_status(manifest: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    graph_rerank_methods = set(get_graph_rerank_methods())
    for split in ("train", "dev", "test"):
        rows.append(_artifact_status(stage="prepare", split=split, path=manifest["artifacts"]["inputs"][split]["input"]))
        rows.append(_artifact_status(stage="graphs", split=split, path=manifest["artifacts"]["graphs"][split]))
    for method in manifest["selected_methods"]:
        if get_method_spec(method).requires_checkpoint:
            learned = manifest["artifacts"]["learned"][method]
            rows.append(_artifact_status(stage="pairs", method=method, path=learned["train_pairs"]))
            rows.append(_artifact_status(stage="train", method=method, path=learned["best_checkpoint"]))
        if method in graph_rerank_methods:
            rows.append(_artifact_status(stage="tune", method=method, path=manifest["artifacts"]["tuned"][method]))
        rows.append(_retrieval_status(manifest, method))
        rows.append(_artifact_status(stage="evaluate", method=method, path=manifest["artifacts"]["metrics"][method]))
    rows.append(_artifact_status(stage="aggregate", path=manifest["artifacts"]["tables"]["main"]))
    return rows


def format_commands(commands: Sequence[StageCommand], *, color: bool = False) -> str:
    return "\n\n".join(_format_command_block(index, command, color=color) for index, command in enumerate(commands, 1))


def format_status(rows: Sequence[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        qualifier = row.get("method") or row.get("split") or ""
        lines.append(f"{row['stage']} {qualifier} {row['state']}".strip())
    return "\n".join(lines)


def list_stage_specs(methods: Sequence[str] | None = None) -> list[dict[str, str]]:
    workflow = set(required_stages_for_methods(methods)) if methods is not None else set(STAGE_ORDER)
    return [
        {
            "name": stage,
            "description": STAGE_DESCRIPTIONS[stage],
            "default": "yes" if stage in workflow else "no",
        }
        for stage in STAGE_ORDER
    ]


def list_method_specs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in get_supported_methods():
        spec = get_method_spec(method)
        rows.append(
            {
                "name": method,
                "workflow": ", ".join(required_stages_for_methods([method])),
                "requires_graphs": str(spec.requires_graphs).lower(),
                "requires_graph_config": str(spec.requires_graph_config).lower(),
                "requires_checkpoint": str(spec.requires_checkpoint).lower(),
                "requires_dense_encoder": str(spec.requires_dense_encoder).lower(),
                "seed_method": spec.seed_method or "",
            }
        )
    return rows


def list_config_entries(kind: str = "all") -> list[ConfigEntry]:
    selected_kinds = _selected_config_kinds(kind)
    entries: list[ConfigEntry] = []
    if "experiments" in selected_kinds:
        entries.extend(_config_entries(EXPERIMENT_CONFIG_DIR, kind="experiment"))
    if "search-spaces" in selected_kinds:
        entries.extend(_config_entries(SEARCH_SPACE_CONFIG_DIR, kind="search-space"))
    if "training" in selected_kinds and TRAINING_CONFIG_DIR.exists():
        for path in sorted(TRAINING_CONFIG_DIR.glob("*/*.json")):
            entries.append(
                ConfigEntry(
                    kind="training",
                    name=f"{path.parent.name}/{path.stem}",
                    path=_path_str(path),
                )
            )
    return entries


def list_profile_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Experiment config profiles must be an object.")
    rows: list[dict[str, Any]] = []
    for name, profile in sorted(profiles.items()):
        if not isinstance(profile, dict):
            raise ValueError(f"Experiment config profile must be an object: {name}")
        effective_config = build_effective_config(config, profile=str(name))
        rows.append(
            {
                "name": str(name),
                "train": str(profile.get("train_examples", "")),
                "dev": str(profile.get("dev_examples", "")),
                "test": str(profile.get("test_examples", "")),
                "splits": {
                    split: {
                        "source": str(split_config["source"]),
                        "max_examples": str(split_config["max_examples"]),
                        "seed": str(split_config["seed"]),
                        "offset": str(split_config["offset"]),
                    }
                    for split, split_config in effective_config["splits"].items()
                },
            }
        )
    return rows


def list_recipe_specs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in list_config_entries("experiments"):
        config = load_experiment_config(entry.path)
        methods = config.get("methods", [])
        rows.append(
            {
                "name": entry.name,
                "recipe": str(config.get("recipe", "")),
                "dataset": str(config.get("dataset", "")),
                "task": str(config.get("task", "")),
                "methods": ", ".join(str(method) for method in methods) if isinstance(methods, list) else "",
                "path": entry.path,
            }
        )
    return rows


def _build_artifact_paths(run_dir: Path, methods: Sequence[str]) -> dict[str, Any]:
    graph_rerank_methods = set(get_graph_rerank_methods())
    inputs = {
        split: {
            "input": _path_str(run_dir / "inputs" / f"{split}.input.json"),
            "labels": _path_str(run_dir / "inputs" / f"{split}.labels.json"),
            "combined": _path_str(run_dir / "inputs" / f"{split}.combined.json"),
        }
        for split in ("train", "dev", "test")
    }
    graphs = {split: _path_str(run_dir / "graphs" / f"{split}.graphs.json") for split in ("train", "dev", "test")}
    tuned = {
        method: _path_str(run_dir / "tuned" / f"{method}.dev_selected.json")
        for method in methods
        if method in graph_rerank_methods
    }
    predictions = {
        method: _path_str(run_dir / "predictions" / f"test.{method}.ranked.json")
        for method in methods
    }
    metrics = {
        method: _path_str(run_dir / "metrics" / f"test.{method}.metrics.csv")
        for method in methods
    }
    failures = {
        method: _path_str(run_dir / "debug" / f"failure_cases_{method}.jsonl")
        for method in methods
    }
    learned = {
        method: {
            "train_pairs": _path_str(run_dir / "learned" / method / "train.pairs.json"),
            "train_pair_summary": _path_str(run_dir / "learned" / method / "train.pairs.summary.json"),
            "train_pair_run_summary": _path_str(run_dir / "learned" / method / "train.pairs.run_summary.json"),
            "effective_training_config": _path_str(run_dir / "learned" / method / "effective_training_config.json"),
            "training_output_dir": _path_str(run_dir / "learned" / method),
            "train_metrics": _path_str(run_dir / "learned" / method / "train_metrics.jsonl"),
            "train_run_summary": _path_str(run_dir / "learned" / method / "train_run_summary.json"),
            "best_checkpoint": _path_str(run_dir / "learned" / method / "checkpoints" / "best.pt"),
        }
        for method in methods
        if get_method_spec(method).requires_checkpoint
    }
    return {
        "inputs": inputs,
        "graphs": graphs,
        "tuned": tuned,
        "learned": learned,
        "predictions": predictions,
        "metrics": metrics,
        "failure_cases": failures,
        "tables": {
            "main": _path_str(run_dir / "tables" / "main_results.csv"),
            "path": _path_str(run_dir / "tables" / "path_results.csv"),
            "efficiency": _path_str(run_dir / "tables" / "efficiency_results.csv"),
        },
    }


def _prepare_commands(manifest: dict[str, Any]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    config = manifest["effective_config"]
    for split in ("train", "dev", "test"):
        split_config = config["splits"][split]
        raw_source = split_config["source"]
        artifacts = manifest["artifacts"]["inputs"][split]
        commands.append(
            StageCommand(
                stage="prepare",
                split=split,
                argv=[
                    sys.executable,
                    "scripts/prepare_hotpotqa.py",
                    "--input",
                    str(config["raw"][raw_source]),
                    "--output_input",
                    artifacts["input"],
                    "--output_labels",
                    artifacts["labels"],
                    "--output_combined",
                    artifacts["combined"],
                    "--max_examples",
                    str(split_config["max_examples"]),
                    "--seed",
                    str(split_config["seed"]),
                    "--offset",
                    str(split_config["offset"]),
                ],
            )
        )
    return commands


def _graph_commands(manifest: dict[str, Any]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    graph_config = manifest["effective_config"]["graph"]
    for split in ("train", "dev", "test"):
        argv = [
            sys.executable,
            "scripts/build_graphs.py",
            "--input",
            manifest["artifacts"]["inputs"][split]["input"],
            "--output",
            manifest["artifacts"]["graphs"][split],
            "--max_query_overlap",
            str(graph_config["max_query_overlap"]),
            "--max_entity_neighbors",
            str(graph_config["max_entity_neighbors"]),
            "--max_bridge_edges",
            str(graph_config["max_bridge_edges"]),
        ]
        if graph_config.get("use_spacy"):
            argv.append("--use_spacy")
        commands.append(StageCommand(stage="graphs", split=split, argv=argv))
    return commands


def _pair_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        if not get_method_spec(method).requires_checkpoint:
            continue
        learned = manifest["artifacts"]["learned"][method]
        commands.append(
            StageCommand(
                stage="pairs",
                method=method,
                argv=[
                    sys.executable,
                    "scripts/build_train_pairs.py",
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
                ],
            )
        )
    return commands


def _train_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        if not get_method_spec(method).requires_checkpoint:
            continue
        learned = manifest["artifacts"]["learned"][method]
        commands.append(
            StageCommand(
                stage="train",
                method=method,
                argv=[
                    sys.executable,
                    "scripts/train_graph_retriever.py",
                    "--train_tasks",
                    manifest["artifacts"]["inputs"]["train"]["input"],
                    "--train_labels",
                    manifest["artifacts"]["inputs"]["train"]["labels"],
                    "--train_graphs",
                    manifest["artifacts"]["graphs"]["train"],
                    "--train_pairs",
                    learned["train_pairs"],
                    "--dev_tasks",
                    manifest["artifacts"]["inputs"]["dev"]["input"],
                    "--dev_labels",
                    manifest["artifacts"]["inputs"]["dev"]["labels"],
                    "--dev_graphs",
                    manifest["artifacts"]["graphs"]["dev"],
                    "--output_dir",
                    learned["training_output_dir"],
                    "--config",
                    learned["effective_training_config"],
                ],
            )
        )
    return commands


def _tune_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    graph_rerank_methods = set(get_graph_rerank_methods())
    for method in methods:
        if method not in graph_rerank_methods:
            continue
        argv = [
            sys.executable,
            "scripts/tune_graph_rerank.py",
            "--method",
            method,
            "--tasks",
            manifest["artifacts"]["inputs"]["dev"]["input"],
            "--labels",
            manifest["artifacts"]["inputs"]["dev"]["labels"],
            "--graphs",
            manifest["artifacts"]["graphs"]["dev"],
            "--output_config",
            manifest["artifacts"]["tuned"][method],
            "--top_k",
            str(manifest["effective_config"]["top_k"]),
            "--grid_config",
            str(manifest["effective_config"]["search_spaces"]["graph_rerank"]),
        ]
        _append_dense_args(argv, manifest)
        commands.append(StageCommand(stage="tune", method=method, argv=argv))
    return commands


def _retrieve_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    graph_rerank_methods = set(get_graph_rerank_methods())
    dense_methods = set(get_methods_requiring_dense_encoder())
    for method in methods:
        spec = get_method_spec(method)
        argv = [
            sys.executable,
            "scripts/run_retrieval.py",
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
        if method in graph_rerank_methods:
            argv.extend(
                [
                    "--graph_config",
                    manifest["artifacts"]["tuned"][method],
                ]
            )
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
        if method in dense_methods and not spec.requires_checkpoint:
            _append_dense_args(argv, manifest)
        commands.append(StageCommand(stage="retrieve", method=method, argv=argv))
    return commands


def _evaluate_commands(manifest: dict[str, Any], methods: Sequence[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for method in methods:
        commands.append(
            StageCommand(
                stage="evaluate",
                method=method,
                argv=[
                    sys.executable,
                    "scripts/evaluate_retrieval.py",
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
                ],
            )
        )
    return commands


def _aggregate_command(manifest: dict[str, Any]) -> StageCommand:
    return StageCommand(
        stage="aggregate",
        argv=[
            sys.executable,
            "scripts/aggregate_tables.py",
            "--input_dir",
            _path_str(Path(manifest["paths"]["run_dir"]) / "metrics"),
            "--output_main",
            manifest["artifacts"]["tables"]["main"],
            "--output_path",
            manifest["artifacts"]["tables"]["path"],
            "--output_efficiency",
            manifest["artifacts"]["tables"]["efficiency"],
        ],
    )


def _append_dense_args(argv: list[str], manifest: dict[str, Any]) -> None:
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


def required_stages_for_methods(methods: Sequence[str]) -> list[str]:
    selected_methods = list(methods)
    _validate_methods(selected_methods)
    required = {"prepare", "graphs", "retrieve", "evaluate", "aggregate"}
    for method in selected_methods:
        spec = get_method_spec(method)
        if spec.requires_graph_config:
            required.add("tune")
        if spec.requires_checkpoint:
            required.update({"pairs", "train"})
    return [stage for stage in STAGE_ORDER if stage in required]


def _select_stages(
    stages: Sequence[str] | None,
    *,
    from_stage: str | None,
    to_stage: str | None,
    methods: Sequence[str] | None = None,
) -> list[str]:
    selected: list[str]
    if stages is not None:
        if from_stage is not None or to_stage is not None:
            raise ValueError("Use either explicit stages or a stage range, not both.")
        selected = list(stages)
    else:
        workflow = required_stages_for_methods(methods) if methods is not None else list(STAGE_ORDER)
        if from_stage is None and to_stage is None:
            selected = workflow
        else:
            start_stage = from_stage or workflow[0]
            end_stage = to_stage or workflow[-1]
            start_index = _workflow_stage_index(workflow, start_stage)
            end_index = _workflow_stage_index(workflow, end_stage)
            if start_index > end_index:
                raise ValueError(f"Stage range start={start_stage} comes after end={end_stage}.")
            selected = workflow[start_index : end_index + 1]
    unknown = [stage for stage in selected if stage not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unsupported stage: {', '.join(unknown)}")
    return selected


def _workflow_stage_index(workflow: Sequence[str], stage: str) -> int:
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unsupported stage: {stage}")
    try:
        return list(workflow).index(stage)
    except ValueError as error:
        raise ValueError(
            f"Stage {stage} is not in the available workflow stages: {', '.join(workflow)}"
        ) from error


def _validate_methods(methods: Iterable[str]) -> None:
    supported_methods = set(get_supported_methods())
    unsupported = [method for method in methods if method not in supported_methods]
    if unsupported:
        raise ValueError(f"Unsupported method: {', '.join(unsupported)}")


def _validate_stage_dependencies(
    manifest: dict[str, Any],
    selected_stages: Sequence[str],
    selected_methods: Sequence[str],
) -> None:
    if "retrieve" in selected_stages and "tune" not in selected_stages:
        graph_rerank_methods = set(get_graph_rerank_methods())
        missing_tuned = [
            method
            for method in selected_methods
            if method in graph_rerank_methods and not Path(manifest["artifacts"]["tuned"][method]).exists()
        ]
        if missing_tuned:
            missing_paths = ", ".join(manifest["artifacts"]["tuned"][method] for method in missing_tuned)
            raise ValueError(
                "Graph rerank retrieval requires tuned graph config. "
                "Add the tune stage or run tune first. "
                f"Missing: {missing_paths}"
            )

    trainable_methods = [method for method in selected_methods if get_method_spec(method).requires_checkpoint]
    if "train" in selected_stages and "pairs" not in selected_stages:
        missing_pairs = [
            method
            for method in trainable_methods
            if not Path(manifest["artifacts"]["learned"][method]["train_pairs"]).exists()
        ]
        if missing_pairs:
            missing_paths = ", ".join(manifest["artifacts"]["learned"][method]["train_pairs"] for method in missing_pairs)
            raise ValueError(
                "Trainable training requires train pairs. "
                "Add the pairs stage or run pairs first. "
                f"Missing: {missing_paths}"
            )

    if "retrieve" in selected_stages and "train" not in selected_stages:
        missing_checkpoints = [
            method
            for method in trainable_methods
            if not Path(manifest["artifacts"]["learned"][method]["best_checkpoint"]).exists()
        ]
        if missing_checkpoints:
            missing_paths = ", ".join(
                manifest["artifacts"]["learned"][method]["best_checkpoint"] for method in missing_checkpoints
            )
            raise ValueError(
                "Trainable retrieval requires a trained checkpoint. "
                "Add the train stage or run train first. "
                f"Missing: {missing_paths}"
            )


def _artifact_status(
    *,
    stage: str,
    path: str,
    method: str | None = None,
    split: str | None = None,
) -> dict[str, str]:
    state = "complete" if Path(path).exists() else "missing"
    row = {"stage": stage, "state": state, "path": path}
    if method is not None:
        row["method"] = method
    if split is not None:
        row["split"] = split
    return row


def _retrieval_status(manifest: dict[str, Any], method: str) -> dict[str, str]:
    path = Path(manifest["artifacts"]["predictions"][method])
    row = {"stage": "retrieve", "method": method, "path": _path_str(path), "state": "missing"}
    if not path.exists():
        return row
    summary_path = path.with_name(f"{path.stem}.run_summary.json")
    if not summary_path.exists():
        row["state"] = "complete"
        return row
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        row["state"] = "stale"
        return row
    expected_tasks = manifest["artifacts"]["inputs"]["test"]["input"]
    expected_top_k = manifest["effective_config"]["top_k"]
    if (
        summary.get("status") == "success"
        and summary.get("inputs", {}).get("tasks") == expected_tasks
        and _same_path(summary.get("outputs", {}).get("predictions"), path)
        and summary.get("effective_config", {}).get("method") == method
        and summary.get("effective_config", {}).get("top_k") == expected_top_k
    ):
        row["state"] = "complete"
    else:
        row["state"] = "stale"
    return row


def _reject_config_change(
    existing: dict[str, Any],
    *,
    config: dict[str, Any],
    profile: str | None,
    methods: Sequence[str] | None,
    cli_overrides: dict[str, Any] | None,
) -> None:
    requested_config = build_effective_config(config, profile=profile, cli_overrides=cli_overrides)
    requested_methods = list(methods) if methods is not None else list(requested_config["methods"])
    _attach_resolved_training_configs(requested_config, requested_methods)
    if requested_config != existing.get("effective_config") or requested_methods != existing.get("selected_methods"):
        raise ValueError("Existing experiment manifest uses a different config; pass force=True to reinitialize.")


def _status_key(row: dict[str, str]) -> str:
    if row.get("method"):
        return f"{row['stage']}:{row['method']}"
    if row.get("split"):
        return f"{row['stage']}:{row['split']}"
    return row["stage"]


def _format_command_block(index: int, command: StageCommand, *, color: bool) -> str:
    qualifier = ""
    if command.method is not None:
        qualifier = f" method={command.method}"
    elif command.split is not None:
        qualifier = f" split={command.split}"
    lines = [
        f"[{index}] {command.stage}{qualifier}",
        f"script: {_command_script(command.argv)}",
        "command:",
    ]
    lines.extend(_format_argv_lines(command.argv, color=color))
    return "\n".join(lines)


def _command_script(argv: Sequence[str]) -> str:
    for value in argv:
        if value.endswith(".py"):
            return value
    return argv[0] if argv else ""


def _format_argv_lines(argv: Sequence[str], *, color: bool) -> list[str]:
    lines: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value.startswith("--"):
            option = _color_option(value, color=color)
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                lines.append(f"  {option} {argv[index + 1]}")
                index += 2
            else:
                lines.append(f"  {option}")
                index += 1
        else:
            lines.append(f"  {value}")
            index += 1
    return lines


def _color_option(value: str, *, color: bool) -> str:
    if not color:
        return value
    return f"\033[36m{value}\033[0m"


def _selected_config_kinds(kind: str) -> set[str]:
    if kind == "all":
        return {"experiments", "search-spaces", "training"}
    if kind in {"experiments", "search-spaces", "training"}:
        return {kind}
    raise ValueError(f"Unsupported config kind: {kind}")


def _config_entries(directory: Path, *, kind: str) -> list[ConfigEntry]:
    if not directory.exists():
        return []
    return [
        ConfigEntry(kind=kind, name=path.stem, path=_path_str(path))
        for path in sorted(directory.glob("*.json"))
    ]


def _looks_like_explicit_path(value: str) -> bool:
    candidate = Path(value)
    return candidate.is_absolute() or "/" in value or "\\" in value


def _looks_like_explicit_training_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute() or normalized.startswith("configs/") or normalized.endswith(".json"):
        return True
    return False


def _path_str(path: str | Path) -> str:
    return Path(path).as_posix()


def _same_path(left: object, right: str | Path) -> bool:
    if not isinstance(left, str):
        return False
    return Path(left) == Path(right)
