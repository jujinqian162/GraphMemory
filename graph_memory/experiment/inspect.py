from __future__ import annotations

from pathlib import Path
from typing import Literal

from graph_memory.experiment.config import ClosedModel
from graph_memory.registry import Registry
from graph_memory.registry.ablations import ABLATION_SUITE_PATCHES

InspectionKind = Literal[
    "methods",
    "datasets",
    "profiles",
    "configs",
    "variants",
    "jobs",
]


class InspectCommandConfig(ClosedModel):
    kind: InspectionKind
    name: str | None = None


def inspect_catalog(
    kind: InspectionKind,
    *,
    repository_root: Path,
    name: str | None = None,
) -> object:
    config_root = repository_root.resolve() / "configs"
    if kind == "methods":
        return [_method_row(method) for method in Registry.methods.list_ids()]
    if kind in {"datasets", "profiles"}:
        directory = config_root / ("dataset" if kind == "datasets" else "profile")
        return sorted(path.stem for path in directory.glob("*.yaml"))
    if kind == "configs":
        return sorted(path.stem for path in config_root.glob("*.yaml"))
    if kind == "variants":
        return {
            method: [
                {
                    "variant": variant.identifier,
                    "changed_dimensions": sorted(variant.changed_dimensions),
                    "baseline_alias": variant.baseline_alias,
                }
                for variant in suite.variants
            ]
            for method, suite in ABLATION_SUITE_PATCHES.items()
        }
    if kind == "jobs":
        if name is None:
            raise ValueError("inspect kind=jobs requires name=<multirun-name>")
        named_root = repository_root.resolve() / "runs" / name
        return sorted(
            path.parent.parent.name
            for path in named_root.glob("*/workflow/summary.yaml")
            if path.is_file()
        )
    raise ValueError(f"unsupported inspection kind: {kind}")


def _method_row(method) -> dict[str, object]:
    spec = Registry.methods.get(method)
    return {
        "method": method.value,
        "lifecycle": spec.lifecycle.value,
        "request_type": spec.input_spec.request_type.__name__,
        "required_artifact": spec.input_spec.required_artifact.value,
        "supported_families": sorted(
            family.value for family in spec.input_spec.supported_families
        ),
        "produces_native_edge_trace": (spec.capabilities.produces_native_edge_trace),
        "train_artifact_kind": (
            None if spec.train_artifact is None else spec.train_artifact.kind.value
        ),
        "train_dependencies": [
            dependency.value for dependency in spec.train_dependencies
        ],
    }
