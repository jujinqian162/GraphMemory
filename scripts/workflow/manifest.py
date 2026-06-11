from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graph_memory.io import merge_config, read_json, write_json
from graph_memory.observability import now_iso
from graph_memory.retrieval_registry import get_method_spec, get_supported_methods
from graph_memory.training_config import load_trainable_training_config
from scripts.workflow.artifacts import build_main_method_artifacts, build_variant_artifact_namespace
from scripts.workflow.registry import ABLATION_SUITE_REGISTRY, get_ablation_suite, get_variant_spec
from scripts.workflow.stage_configs import attach_stage_config_projections
from scripts.workflow.types import ArtifactRole, ConfigEntry, StageId, VariantArtifactNamespace, VariantSpec

CONFIG_ROOT = Path("configs")
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
SEARCH_SPACE_CONFIG_DIR = CONFIG_ROOT / "search_spaces"
TRAINING_CONFIG_DIR = CONFIG_ROOT / "training"
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/hotpotqa_evidence_retrieval.json")
DEFAULT_SEARCH_SPACE_CONFIG = Path("configs/search_spaces/graph_rerank.json")
STAGE_DESCRIPTIONS = {
    StageId.PREPARE.value: "Build split-specific task, label, and combined input artifacts.",
    StageId.GRAPHS.value: "Build evidence graph artifacts for train, dev, and test splits.",
    StageId.PAIRS.value: "Build supervised training pairs for checkpoint-backed methods.",
    StageId.TUNE.value: "Select graph-rerank parameters from the search-space config.",
    StageId.TRAIN.value: "Train checkpoint-backed graph retrievers.",
    StageId.RETRIEVE.value: "Write ranked evidence predictions for selected methods.",
    StageId.EVALUATE.value: "Evaluate ranked predictions and write per-method metrics.",
    StageId.AGGREGATE.value: "Aggregate per-method metrics into report tables.",
}


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
    """Initialize a schema-v2 workflow manifest with optional ablation units."""

    from scripts.workflow.planner import _select_stages, _validate_methods

    run_dir = Path(run_root) / experiment_name
    manifest_path = run_dir / "manifest.json"
    enabled = _ablation_enabled(config)
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        if not isinstance(existing, dict):
            raise ValueError(f"Existing manifest must be a JSON object: {manifest_path}")
        if existing.get("schema_version", 1) == 1 and enabled:
            raise ValueError("Schema-version-1 manifest cannot enable ablation in place; reinitialize with force=True.")
        _reject_config_change(existing, config=config, profile=profile, methods=methods, cli_overrides=cli_overrides)
        return existing

    effective_config = build_effective_config(config, profile=profile, cli_overrides=cli_overrides)
    selected_methods = list(methods) if methods is not None else list(effective_config["methods"])
    _validate_methods(selected_methods)
    selected_variants = _selected_suite_variants(config, selected_methods) if enabled else {}
    _attach_resolved_training_configs(effective_config, selected_methods)
    selected_stages = _select_stages(stages, from_stage=None, to_stage=None, methods=selected_methods)
    manifest: dict[str, Any] = {
        "schema_version": 2,
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
    _write_resolved_training_configs(manifest)
    attach_stage_config_projections(manifest)
    if enabled:
        _attach_ablation_artifacts(manifest, selected_variants)
    else:
        manifest["artifacts"]["ablations"] = {}
    write_json(manifest["paths"]["effective_config"], manifest["effective_config"])
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
        "enable_ablation": _ablation_enabled(config),
        "ablation_variants": config.get("ablation_variants", {}),
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
            entries.append(ConfigEntry(kind="training", name=f"{path.parent.name}/{path.stem}", path=_path_str(path)))
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


def _attach_resolved_training_configs(effective_config: dict[str, Any], selected_methods: Sequence[str]) -> None:
    from scripts.workflow.planner import _method_has_stage

    training_config_paths = effective_config.get("training_configs", {})
    if not isinstance(training_config_paths, dict):
        raise ValueError("Experiment config training_configs must be an object.")
    resolved_by_method: dict[str, Any] = {}
    for method in selected_methods:
        if not _method_has_stage(method, StageId.TRAIN):
            continue
        config_path = training_config_paths.get(method)
        if not isinstance(config_path, str) or not config_path:
            raise ValueError(f"Trainable method={method} requires a training config path.")
        resolved = load_trainable_training_config(
            resolve_training_config_path(method, config_path),
            profile=str(effective_config["profile"]),
        )
        if resolved["method"] != method:
            raise ValueError(f"Training config method={resolved['method']} does not match selected method={method}.")
        resolved_by_method[method] = resolved
    if resolved_by_method:
        effective_config["training"] = resolved_by_method


def _write_resolved_training_configs(manifest: dict[str, Any]) -> None:
    training_configs = manifest["effective_config"].get("training", {})
    if not isinstance(training_configs, dict):
        return
    for method, training_config in training_configs.items():
        write_json(manifest["artifacts"]["learned"][method]["effective_training_config"], training_config)


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
            "effective_training_config": main[ArtifactRole.EFFECTIVE_TRAINING_CONFIG],
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
) -> None:
    run_dir = Path(manifest["paths"]["run_dir"])
    index_path = run_dir / "config" / "ablation_metrics_index.json"
    table_path = run_dir / "tables" / "ablation_results.csv"
    metric_entries: list[dict[str, str]] = []
    suite_metadata: dict[str, Any] = {}
    manifest["artifacts"]["ablations"] = {}

    for method, variants in selected_variants.items():
        main_artifacts = build_main_method_artifacts(run_dir, method)
        main_training_config = manifest["effective_config"]["training"][method]
        method_artifacts: dict[str, Any] = {}
        suite_metadata[method] = [variant.identifier.value for variant in variants]
        for variant in variants:
            namespace = build_variant_artifact_namespace(run_dir, method, variant, main_artifacts)
            method_artifacts[variant.identifier.value] = _namespace_record(namespace)
            manifest["run_units"].append({"method": method, "variant": variant.identifier.value})
            metric_entries.append(
                {
                    "method": method,
                    "variant": variant.identifier.value,
                    "metrics_path": namespace.path(ArtifactRole.METRICS),
                }
            )
            if not variant.baseline_alias:
                effective = merge_config(dict(main_training_config), None, dict(variant.training_config_override))
                write_json(namespace.local_paths[ArtifactRole.EFFECTIVE_TRAINING_CONFIG], effective)
        manifest["artifacts"]["ablations"][method] = method_artifacts

    manifest["ablation_suites"] = suite_metadata
    manifest["paths"]["ablation_metrics_index"] = _path_str(index_path)
    manifest["artifacts"]["tables"]["ablation"] = _path_str(table_path)
    write_json(index_path, {"schema_version": 1, "metrics": metric_entries})


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
    _attach_resolved_training_configs(requested_config, requested_methods)
    if existing.get("schema_version", 1) == 1:
        requested_config.pop("enable_ablation", None)
        requested_config.pop("ablation_variants", None)
    if requested_config != existing.get("effective_config") or requested_methods != existing.get("selected_methods"):
        raise ValueError("Existing experiment manifest uses a different config; pass force=True to reinitialize.")


def _selected_config_kinds(kind: str) -> set[str]:
    if kind == "all":
        return {"experiments", "search-spaces", "training"}
    if kind in {"experiments", "search-spaces", "training"}:
        return {kind}
    raise ValueError(f"Unsupported config kind: {kind}")


def _config_entries(directory: Path, *, kind: str) -> list[ConfigEntry]:
    if not directory.exists():
        return []
    return [ConfigEntry(kind=kind, name=path.stem, path=_path_str(path)) for path in sorted(directory.glob("*.json"))]


def _looks_like_explicit_path(value: str) -> bool:
    candidate = Path(value)
    return candidate.is_absolute() or "/" in value or "\\" in value


def _looks_like_explicit_training_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    candidate = Path(value)
    return candidate.is_absolute() or normalized.startswith("configs/") or normalized.endswith(".json")


def _path_str(path: str | Path) -> str:
    return Path(path).as_posix()
