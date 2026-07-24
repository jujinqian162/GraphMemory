from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import cast


from graph_memory.evaluation.tables import (
    EFFICIENCY_RESULT_COLUMNS,
    MAIN_RESULT_COLUMNS,
    PATH_RESULT_COLUMNS,
)
from graph_memory.experiment.artifacts import artifact_payload_path
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
            "benchmark": (
                None
                if result.benchmark is None
                else result.benchmark.model_dump(mode="json")
            ),
        },
    )
    write_yaml_atomic(
        destination / "assets" / "manifest.yaml",
        {"assets": [asset.model_dump(mode="json") for asset in result.assets]},
    )

    metric_rows = [dict(row) for row in result.evaluation.metric_rows]
    if len(metric_rows) != 1:
        raise ValueError(
            f"one final method requires exactly one metric row, got {len(metric_rows)}"
        )
    final_row = cast(dict[str, object], metric_rows[0])
    if result.benchmark is None:
        final_row["Retrieval Latency / Query"] = "NA"
    else:
        final_row["Retrieval Latency / Query"] = result.benchmark.metrics[
            "benchmark.retrieval_latency_ms_per_query"
        ]
    final_row["Method"] = result.method
    if result.variant is not None:
        final_row["Variant"] = result.variant

    fields = list(metric_rows[0])
    if result.variant is not None and "Variant" not in fields:
        fields.insert(1, "Variant")
    _write_rows(destination / "metrics" / "final.metrics.csv", [final_row], fields)
    _write_rows(
        destination / "tables" / "main_results.csv",
        [_select(final_row, MAIN_RESULT_COLUMNS, result.variant)],
        _fields(MAIN_RESULT_COLUMNS, result.variant),
    )
    _write_rows(
        destination / "tables" / "path_results.csv",
        [_select(final_row, PATH_RESULT_COLUMNS, result.variant)],
        _fields(PATH_RESULT_COLUMNS, result.variant),
    )
    _write_rows(
        destination / "tables" / "efficiency_results.csv",
        [_select(final_row, EFFICIENCY_RESULT_COLUMNS, result.variant)],
        _fields(EFFICIENCY_RESULT_COLUMNS, result.variant),
    )

    if result.model is not None:
        history = artifact_payload_path(result.model.artifact, "training_metrics")
        target = destination / "training" / "train_metrics.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history, target)
        write_yaml_atomic(
            destination / "training" / "origin.yaml",
            {
                "asset": result.model.artifact.model_dump(mode="json"),
            },
        )
    failure_cases = artifact_payload_path(result.evaluation.artifact, "failure_cases")
    debug_target = destination / "debug" / "failure_cases.jsonl"
    debug_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(failure_cases, debug_target)
    per_task = artifact_payload_path(result.evaluation.artifact, "per_task")
    per_task_target = destination / "metrics" / "per_task.jsonl"
    per_task_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(per_task, per_task_target)
    write_yaml_atomic(
        destination / "workflow" / "ranking_origin.yaml",
        {
            "production_seconds": result.ranking.production_seconds,
            "current_runtime_logged": result.benchmark is not None,
        },
    )


def resolved_overrides() -> tuple[str, ...]:
    try:
        from hydra.core.hydra_config import HydraConfig

        overrides = HydraConfig.get().overrides.task
    except (ValueError, AttributeError):
        return ()
    return tuple(str(value) for value in overrides)


def _fields(columns: list[str], variant: str | None) -> list[str]:
    fields = list(columns)
    if variant is not None:
        fields.insert(1, "Variant")
    return fields


def _select(
    row: dict[str, object],
    columns: list[str],
    variant: str | None,
) -> dict[str, object]:
    selected = {column: row.get(column, "NA") for column in columns}
    if variant is not None:
        selected["Variant"] = variant
    return selected


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
