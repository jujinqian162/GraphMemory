from __future__ import annotations

import csv
import shutil
from pathlib import Path

from graph_memory.experiment.artifacts import (
    artifact_csv_rows,
    artifact_payload_path,
    artifact_shape_count,
    prediction_production_seconds,
)
from graph_memory.experiment.config import ResolvedExperimentConfig
from graph_memory.experiment.persistence import write_yaml_atomic
from graph_memory.experiment.results import FinalExperimentResult


def project_run_output(
    output_dir: Path,
    *,
    repository_root: Path,
    config: ResolvedExperimentConfig,
    overrides: tuple[str, ...],
    result: FinalExperimentResult,
) -> None:
    destination = output_dir.resolve()
    runs_root = (repository_root.resolve() / "runs").resolve()
    if destination == runs_root or not destination.is_relative_to(runs_root):
        raise ValueError(f"run output must be a named child below runs/: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(destination / "config" / "resolved.yaml", config.normalized())
    write_yaml_atomic(
        destination / "config" / "overrides.yaml",
        {"overrides": list(overrides)},
    )
    write_yaml_atomic(
        destination / "workflow" / "summary.yaml",
        {
            "method": result.method,
            "variant": result.variant,
            "dataset": config.dataset.name,
            "profile": config.profile,
            "seed": config.seed,
            "cache_refresh": config.cache.refresh,
        },
    )
    write_yaml_atomic(
        destination / "assets" / "manifest.yaml",
        {"assets": [asset.model_dump(mode="json") for asset in result.assets]},
    )

    metric_rows = artifact_csv_rows(result.evaluation, "metrics")
    if len(metric_rows) != 1:
        raise ValueError(
            f"one final method requires exactly one metric row, got {len(metric_rows)}"
        )
    final_row: dict[str, object] = dict(metric_rows[0])
    final_row["Retrieval Latency / Query"] = (
        prediction_production_seconds(result.ranking)
        / max(1, artifact_shape_count(result.evaluation, "per_task_rows"))
    )
    final_row["Method"] = result.method
    if result.variant is not None:
        final_row["Variant"] = result.variant

    fields = list(final_row)
    if result.variant is not None and "Variant" not in fields:
        fields.insert(1, "Variant")
    _write_rows(destination / "metrics" / "final.metrics.csv", [final_row], fields)

    if result.model is not None:
        history = artifact_payload_path(result.model, "training_metrics")
        target = destination / "training" / "train_metrics.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history, target)
        write_yaml_atomic(
            destination / "training" / "origin.yaml",
            {
                "asset": result.model.model_dump(mode="json"),
            },
        )
    failure_cases = artifact_payload_path(result.evaluation, "failure_cases")
    debug_target = destination / "debug" / "failure_cases.jsonl"
    debug_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(failure_cases, debug_target)
    per_task = artifact_payload_path(result.evaluation, "per_task")
    per_task_target = destination / "metrics" / "per_task.jsonl"
    per_task_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(per_task, per_task_target)
    write_yaml_atomic(
        destination / "workflow" / "ranking_origin.yaml",
        {
            "production_seconds": prediction_production_seconds(result.ranking),
            "current_runtime_logged": True,
        },
    )


def resolved_overrides() -> tuple[str, ...]:
    try:
        from hydra.core.hydra_config import HydraConfig

        overrides = HydraConfig.get().overrides.task
    except (ValueError, AttributeError):
        return ()
    return tuple(str(value) for value in overrides)


def _write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "project_run_output",
    "resolved_overrides",
]
