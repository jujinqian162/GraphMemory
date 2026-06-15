from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
import shutil
from typing import Any

from graph_memory.config import CONFIG_LOADER
from graph_memory.config.converter import ConfigConverter
from graph_memory.config.patches import deep_merge_patch
from graph_memory.io import merge_config, read_json, write_json
from graph_memory.observability import now_iso
from graph_memory.registry import Registry
from graph_memory.registry.method_configs import TrainableMethodConfig
from scripts.workflow.artifacts import build_main_method_artifacts, build_variant_artifact_namespace
from scripts.workflow.contracts import validate_current_manifest
from scripts.workflow.registry import (
    ABLATION_SUITE_REGISTRY,
    get_ablation_suite,
    get_variant_spec,
)
from scripts.workflow.stage_configs import (
    load_trainable_method_configs,
    write_main_stage_configs,
    write_variant_stage_configs,
    _memory_stream_importance_path,
)
from scripts.workflow.types import ArtifactRole, ConfigEntry, StageId, VariantArtifactNamespace, VariantSpec

CONFIG_ROOT = Path("configs")
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
SEARCH_SPACE_CONFIG_DIR = CONFIG_ROOT / "search_spaces"
METHOD_CONFIG_DIR = CONFIG_ROOT / "methods"
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/hotpotqa_evidence_retrieval.json")
DEFAULT_GRAPH_RERANK_SEARCH_SPACE_CONFIG = Path(
    "configs/search_spaces/graph_rerank.json"
)
DEFAULT_MEMORY_STREAM_SEARCH_SPACE_CONFIG = Path(
    "configs/search_spaces/memory_stream.json"
)
LOGGER = logging.getLogger(__name__)
STAGE_DESCRIPTIONS = {
    StageId.PREPARE.value: "Build split-specific task, label, and combined input artifacts.",
    StageId.GRAPHS.value: "Build evidence graph artifacts for train, dev, and test splits.",
    StageId.PAIRS.value: "Build supervised training pairs for checkpoint-backed methods.",
    StageId.TUNE.value: "Select graph-rerank parameters from the search-space config.",
    StageId.TRAIN.value: "Train checkpoint-backed retrieval methods.",
    StageId.RETRIEVE.value: "Write ranked evidence predictions for selected methods.",
    StageId.EVALUATE.value: "Evaluate ranked predictions and write per-method metrics.",
    StageId.AGGREGATE.value: "Aggregate per-method metrics into report tables.",
}


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_experiment_config_path(path)
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a JSON object: {config_path}")
    _reject_retired_experiment_config_fields(config)
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


def resolve_method_config_path(path: str | Path) -> Path:
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Method config does not exist: {config_path}")
    return config_path


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
    """Initialize the current workflow manifest with optional ablation units."""

    from scripts.workflow.planner import _select_stages, _validate_methods

    run_dir = Path(run_root) / experiment_name
    manifest_path = run_dir / "manifest.json"
    _reject_retired_experiment_config_fields(config)
    enabled = _ablation_enabled(config)
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        if not isinstance(existing, dict):
            raise ValueError(f"Existing manifest must be a JSON object: {manifest_path}")
        validate_current_manifest(existing)
        _reject_config_change(existing, config=config, profile=profile, methods=methods, cli_overrides=cli_overrides)
        return existing
    if force and run_dir.exists():
        shutil.rmtree(run_dir)

    effective_config = build_effective_config(config, profile=profile, cli_overrides=cli_overrides)
    selected_methods = list(methods) if methods is not None else list(effective_config["methods"])
    _validate_methods(selected_methods)
    selected_variants = _selected_suite_variants(config, selected_methods) if enabled else {}
    method_configs = load_trainable_method_configs(effective_config, selected_methods)
    effective_config["resolved_method_configs"] = {
        method: CONFIG_LOADER.to_json(method_config)
        for method, method_config in method_configs.items()
    }
    selected_stages = _select_stages(stages, from_stage=None, to_stage=None, methods=selected_methods)
    manifest: dict[str, Any] = {
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
        "artifacts": _build_artifact_paths(run_dir, selected_methods),
        "run_units": [{"method": method, "variant": None} for method in selected_methods],
        "stage_status": {},
    }
    manifest = _apply_memory_stream_test_cap(manifest)
    _write_resolved_method_configs(manifest)
    write_main_stage_configs(manifest, method_configs)
    if enabled:
        _attach_ablation_artifacts(manifest, selected_variants, method_configs)
    else:
        manifest["artifacts"]["ablations"] = {}
    write_json(manifest["paths"]["effective_config"], manifest["effective_config"])
    validate_current_manifest(manifest)
    write_json(manifest_path, manifest)
    return manifest


def load_manifest(experiment_name: str, *, run_root: str | Path = "runs") -> dict[str, Any]:
    manifest_path = Path(run_root) / experiment_name / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    validate_current_manifest(manifest)
    return manifest


def build_effective_config(
    config: dict[str, Any],
    *,
    profile: str | None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _reject_retired_experiment_config_fields(config)
    profile_name = profile or str(config.get("default_profile", "quick"))
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile: {profile_name}")

    defaults = dict(config.get("defaults", {}))
    profile_config = dict(profiles[profile_name])
    split_offsets = config.get("split_offsets", {})
    split_sources = config.get("split_sources", {"train": "train", "dev": "dev", "test": "dev"})
    search_spaces = {
        "graph_rerank": str(DEFAULT_GRAPH_RERANK_SEARCH_SPACE_CONFIG),
        "memory_stream": str(DEFAULT_MEMORY_STREAM_SEARCH_SPACE_CONFIG),
    }
    search_spaces.update(config.get("search_spaces", {}))
    base_config = {
        "recipe": config.get("recipe", "hotpotqa_evidence_retrieval"),
        "dataset": config.get("dataset", "hotpotqa"),
        "task": config.get("task", "evidence_retrieval"),
        "profile": profile_name,
        "enable_ablation": _ablation_enabled(config),
        "ablation_variants": config.get("ablation_variants", {}),
        "raw": config["raw"],
        "graph": config.get("graph", {}),
        "methods": config.get("methods", [method.value for method in Registry.methods.list_ids()]),
        "search_spaces": search_spaces,
        "method_configs": config.get("method_configs", {}),
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
    for key in (
        "memory_stream_importance_path",
        "memory_stream_relevance_weight",
        "memory_stream_recency_weight",
        "memory_stream_importance_weight",
        "memory_stream_recency_decay",
    ):
        if key in config:
            base_config[key] = config[key]
    return merge_config(base_config, None, cli_overrides)


def _reject_retired_experiment_config_fields(config: dict[str, Any]) -> None:
    if "training_configs" in config:
        raise ValueError("Experiment config contains retired field: training_configs")


def list_stage_specs(methods: Sequence[str] | None = None) -> list[dict[str, str]]:
    from scripts.workflow.planner import required_stages_for_methods

    workflow = set(required_stages_for_methods(methods)) if methods is not None else {stage.value for stage in StageId}
    return [
        {
            "name": stage.value,
            "description": STAGE_DESCRIPTIONS[stage.value],
            "default": "yes" if stage.value in workflow else "no",
        }
        for stage in StageId
    ]


def list_method_specs() -> list[dict[str, str]]:
    from scripts.workflow.planner import required_stages_for_methods

    rows: list[dict[str, str]] = []
    for method_id in Registry.methods.list_ids():
        method = method_id.value
        definition = Registry.methods.get(method_id)
        rows.append(
            {
                "name": method,
                "workflow": ", ".join(required_stages_for_methods([method])),
                "lifecycle": definition.lifecycle.value,
                "tuning": (
                    definition.tuning.value
                    if definition.tuning is not None
                    else ""
                ),
                "graph_source": definition.dependencies.graphs.value,
                "selected_config_source": definition.dependencies.selected_config.value,
                "model_source": definition.dependencies.model.value,
                "encoder_source": definition.dependencies.encoder.value,
                "seed_method": definition.seed_method.value if definition.seed_method is not None else "",
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
    if "methods" in selected_kinds:
        entries.extend(_config_entries(METHOD_CONFIG_DIR, kind="method"))
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


def _write_resolved_method_configs(manifest: dict[str, Any]) -> None:
    method_configs = manifest["effective_config"].get("resolved_method_configs", {})
    if not isinstance(method_configs, dict):
        return
    for method, method_config in method_configs.items():
        write_json(manifest["artifacts"]["learned"][method]["effective_method_config"], method_config)


def _build_artifact_paths(run_dir: Path, methods: Sequence[str]) -> dict[str, Any]:
    from scripts.workflow.planner import _method_has_stage

    inputs = {
        split: {
            "input": _path_str(run_dir / "inputs" / f"{split}.input.json"),
            "labels": _path_str(run_dir / "inputs" / f"{split}.labels.json"),
            "combined": _path_str(run_dir / "inputs" / f"{split}.combined.json"),
        }
        for split in ("train", "dev", "test")
    }
    learned = {}
    for method in methods:
        if not _method_has_stage(method, StageId.TRAIN):
            continue
        main = build_main_method_artifacts(run_dir, method)
        learned[method] = {
            "train_pairs": main[ArtifactRole.TRAIN_PAIRS],
            "train_pair_summary": main[ArtifactRole.TRAIN_PAIR_SUMMARY],
            "train_pair_run_summary": main[ArtifactRole.TRAIN_PAIR_RUN_SUMMARY],
            "effective_method_config": main[ArtifactRole.EFFECTIVE_METHOD_CONFIG],
            "training_output_dir": _path_str(run_dir / "learned" / method),
            "train_metrics": main[ArtifactRole.TRAIN_METRICS],
            "train_run_summary": main[ArtifactRole.TRAIN_RUN_SUMMARY],
            "best_checkpoint": main[ArtifactRole.CHECKPOINT],
        }
    return {
        "inputs": inputs,
        "graphs": {split: _path_str(run_dir / "graphs" / f"{split}.graphs.json") for split in ("train", "dev", "test")},
        "tuned": {
            method: _path_str(run_dir / "tuned" / f"{method}.dev_selected.json")
            for method in methods
            if _method_has_stage(method, StageId.TUNE)
        },
        "learned": learned,
        "predictions": {method: _path_str(run_dir / "predictions" / f"test.{method}.ranked.json") for method in methods},
        "metrics": {method: _path_str(run_dir / "metrics" / f"test.{method}.metrics.csv") for method in methods},
        "failure_cases": {method: _path_str(run_dir / "debug" / f"failure_cases_{method}.jsonl") for method in methods},
        "tables": {
            "main": _path_str(run_dir / "tables" / "main_results.csv"),
            "path": _path_str(run_dir / "tables" / "path_results.csv"),
            "efficiency": _path_str(run_dir / "tables" / "efficiency_results.csv"),
        },
    }


def _apply_memory_stream_test_cap(manifest: dict[str, Any]) -> dict[str, Any]:
    if "memory_stream" not in manifest.get("selected_methods", []):
        return manifest
    importance_path = _memory_stream_importance_path(manifest, "memory_stream")
    if not importance_path.is_file():
        raise ValueError(f"Memory Stream importance artifact not found: {importance_path}")
    artifact = read_json(importance_path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Memory Stream importance artifact must be a JSON object: {importance_path}")
    tasks = artifact.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Memory Stream importance artifact tasks must be a list: {importance_path}")
    coverage = len(tasks)
    test_split = manifest["effective_config"]["splits"]["test"]
    requested_count = int(test_split["max_examples"])
    if requested_count <= coverage:
        return manifest
    test_split["max_examples"] = coverage
    LOGGER.warning(
        "Memory Stream test split capped from %s to %s using %s",
        requested_count,
        coverage,
        importance_path,
    )
    return manifest


def _ablation_enabled(config: dict[str, Any]) -> bool:
    enabled = config.get("enable_ablation", False)
    if not isinstance(enabled, bool):
        raise ValueError("Experiment config enable_ablation must be a boolean.")
    return enabled


def _selected_suite_variants(config: dict[str, Any], selected_methods: Sequence[str]) -> dict[str, tuple[VariantSpec, ...]]:
    configured = config.get("ablation_variants", {})
    if not isinstance(configured, dict):
        raise ValueError("Experiment config ablation_variants must be an object.")
    unknown_methods = sorted(set(configured) - set(ABLATION_SUITE_REGISTRY))
    if unknown_methods:
        allowed = ", ".join(sorted(ABLATION_SUITE_REGISTRY))
        raise ValueError(f"Unknown ablation suite methods={unknown_methods}; allowed values: {allowed}")

    selected: dict[str, tuple[VariantSpec, ...]] = {}
    for method in selected_methods:
        suite = get_ablation_suite(method)
        if suite is None:
            continue
        requested = configured.get(method)
        if requested is None:
            selected[method] = suite.variants
            continue
        if not isinstance(requested, list) or not all(isinstance(value, str) for value in requested):
            raise ValueError(f"Experiment config ablation_variants[{method}] must be a list of strings.")
        baseline = _baseline_variant(suite.variants, method=method)
        variants = [baseline]
        for value in requested:
            variant = get_variant_spec(method, value)
            if variant not in variants:
                variants.append(variant)
        selected[method] = tuple(variants)
    return selected


def _baseline_variant(variants: Sequence[VariantSpec], *, method: str) -> VariantSpec:
    baseline_aliases = [variant for variant in variants if variant.baseline_alias]
    if len(baseline_aliases) != 1:
        raise ValueError(f"Ablation suite method={method} must declare exactly one baseline alias.")
    return baseline_aliases[0]


def _attach_ablation_artifacts(
    manifest: dict[str, Any],
    selected_variants: dict[str, tuple[VariantSpec, ...]],
    method_configs: dict[str, TrainableMethodConfig],
) -> None:
    run_dir = Path(manifest["paths"]["run_dir"])
    index_path = run_dir / "config" / "ablation_metrics_index.json"
    table_path = run_dir / "tables" / "ablation_results.csv"
    metric_entries: list[dict[str, str]] = []
    suite_metadata: dict[str, Any] = {}
    manifest["artifacts"]["ablations"] = {}

    for method, variants in selected_variants.items():
        main_artifacts = build_main_method_artifacts(run_dir, method)
        main_method_config = method_configs[method]
        main_method_config_record = CONFIG_LOADER.to_json(main_method_config)
        if not isinstance(main_method_config_record, dict):
            raise ValueError(f"Method config must serialize to an object: {method}")
        method_artifacts: dict[str, Any] = {}
        suite_metadata[method] = [variant.identifier.value for variant in variants]
        for variant in variants:
            namespace = build_variant_artifact_namespace(run_dir, method, variant, main_artifacts)
            record = _namespace_record(namespace)
            method_artifacts[variant.identifier.value] = record
            manifest["run_units"].append({"method": method, "variant": variant.identifier.value})
            metric_entries.append(
                {
                    "method": method,
                    "variant": variant.identifier.value,
                    "metrics_path": namespace.path(ArtifactRole.METRICS),
                }
            )
            if variant.baseline_alias:
                record["stage_configs"] = {
                    stage: paths[method]
                    for stage, paths in manifest["stage_configs"].items()
                    if method in paths
                }
                continue
            effective = deep_merge_patch(main_method_config_record, dict(variant.training_config_override))
            typed_effective = ConfigConverter().structure(effective, TrainableMethodConfig)
            write_json(namespace.local_paths[ArtifactRole.EFFECTIVE_METHOD_CONFIG], effective)
            record["stage_configs"] = write_variant_stage_configs(
                manifest,
                method=method,
                variant=variant.identifier.value,
                method_config=typed_effective,
                record=record,
            )
        manifest["artifacts"]["ablations"][method] = method_artifacts

    manifest["ablation_suites"] = suite_metadata
    manifest["paths"]["ablation_metrics_index"] = _path_str(index_path)
    manifest["artifacts"]["tables"]["ablation"] = _path_str(table_path)
    write_json(index_path, {"metrics": metric_entries})


def _namespace_record(namespace: VariantArtifactNamespace) -> dict[str, Any]:
    return {
        "variant": namespace.variant,
        "baseline_alias": namespace.invalidated_from is None,
        "invalidated_from": namespace.invalidated_from.value if namespace.invalidated_from is not None else None,
        "artifacts": {role.value: path for role, path in namespace.paths.items()},
        "aliases": [{"role": alias.role.value, "source": alias.source, "target": alias.target} for alias in namespace.aliases],
    }


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
    method_configs = load_trainable_method_configs(requested_config, requested_methods)
    requested_config["resolved_method_configs"] = {
        method: CONFIG_LOADER.to_json(method_config)
        for method, method_config in method_configs.items()
    }
    if requested_config != existing.get("effective_config") or requested_methods != existing.get("selected_methods"):
        raise ValueError("Existing experiment manifest uses a different config; pass force=True to reinitialize.")


def _selected_config_kinds(kind: str) -> set[str]:
    if kind == "all":
        return {"experiments", "search-spaces", "methods"}
    if kind in {"experiments", "search-spaces", "methods"}:
        return {kind}
    raise ValueError(f"Unsupported config kind: {kind}")


def _config_entries(directory: Path, *, kind: str) -> list[ConfigEntry]:
    if not directory.exists():
        return []
    return [ConfigEntry(kind=kind, name=path.stem, path=_path_str(path)) for path in sorted(directory.glob("*.json"))]


def _looks_like_explicit_path(value: str) -> bool:
    candidate = Path(value)
    return candidate.is_absolute() or "/" in value or "\\" in value


def _path_str(path: str | Path) -> str:
    return Path(path).as_posix()
