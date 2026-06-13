from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph_memory.config.converter import ConfigConverter
from graph_memory.registry import Registry


@dataclass(frozen=True)
class CurrentWorkflowManifest:
    experiment_name: str
    recipe: str
    profile: str | None
    created_at: str
    updated_at: str
    paths: dict[str, Any]
    selected_methods: list[str]
    selected_stages: list[str]
    effective_config: dict[str, Any]
    artifacts: dict[str, Any]
    run_units: list[dict[str, Any]]
    stage_status: dict[str, Any]
    stage_configs: dict[str, dict[str, str]]
    ablation_suites: dict[str, list[str]] = field(default_factory=dict)


def validate_current_manifest(value: object) -> None:
    manifest = ConfigConverter().structure(value, CurrentWorkflowManifest)
    required_stage_mappings = {"pairs", "train", "retrieve", "evaluate"}
    actual_stage_mappings = set(manifest.stage_configs)
    if actual_stage_mappings != required_stage_mappings:
        missing = sorted(required_stage_mappings - actual_stage_mappings)
        extra = sorted(actual_stage_mappings - required_stage_mappings)
        raise ValueError(f"Manifest stage_configs mismatch: missing={missing}, extra={extra}")
    selected_methods = set(manifest.selected_methods)
    trainable_methods = {
        method
        for method in selected_methods
        if Registry.methods.get(method).train_artifact is not None
    }
    expected_methods_by_stage = {
        "pairs": trainable_methods,
        "train": trainable_methods,
        "retrieve": selected_methods,
        "evaluate": selected_methods,
    }
    for stage, expected_methods in expected_methods_by_stage.items():
        actual_methods = set(manifest.stage_configs[stage])
        missing_methods = sorted(expected_methods - actual_methods)
        extra_methods = sorted(actual_methods - expected_methods)
        if missing_methods or extra_methods:
            raise ValueError(
                f"Manifest stage_configs.{stage} is missing selected methods: {missing_methods}; "
                f"unexpected methods: {extra_methods}"
            )
    for stage, paths in manifest.stage_configs.items():
        invalid = sorted(method for method, path in paths.items() if not isinstance(path, str) or not path)
        if invalid:
            raise ValueError(f"Manifest stage_configs.{stage} contains invalid paths for: {invalid}")


__all__ = ["CurrentWorkflowManifest", "validate_current_manifest"]
