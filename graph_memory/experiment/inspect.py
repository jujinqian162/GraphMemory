from __future__ import annotations

from pathlib import Path
from typing import Literal

from graph_memory.experiment.config import ClosedModel
from graph_memory.experiment.planning import STAGE_ORDER
from graph_memory.experiment.registry import METHODS
from graph_memory.registry.ablations import ABLATION_SUITE_PATCHES

InspectionKind = Literal[
    "stages",
    "methods",
    "datasets",
    "profiles",
    "configs",
    "ablations",
]


class InspectCommandConfig(ClosedModel):
    kind: InspectionKind


def inspect_catalog(kind: InspectionKind, *, repository_root: Path) -> object:
    config_root = repository_root.resolve() / "configs"
    if kind == "stages":
        return list(STAGE_ORDER)
    if kind == "methods":
        return [_method_row(method) for method in METHODS.list_ids()]
    if kind in {"datasets", "profiles"}:
        directory = config_root / ("dataset" if kind == "datasets" else "profile")
        return sorted(path.stem for path in directory.glob("*.yaml"))
    if kind == "configs":
        return sorted(path.stem for path in config_root.glob("*.yaml"))
    if kind == "ablations":
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
    raise ValueError(f"unsupported inspection kind: {kind}")


def _method_row(method) -> dict[str, object]:
    spec = METHODS.get(method)
    return {
        "method": method.value,
        "lifecycle": spec.lifecycle.value,
        "tuning": (None if spec.tuning is None else spec.tuning.value),
        "train_artifact_kind": (
            None if spec.train_artifact_kind is None else spec.train_artifact_kind.value
        ),
        "train_dependencies": [
            dependency.value for dependency in spec.train_dependencies
        ],
    }

