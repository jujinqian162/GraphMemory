from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.analysis import analyze_main_results
from graph_memory.io import write_json

# Per-query metrics that may enter the main table or its fixed-k diagnostics.
# Aggregate-only graph/path metrics and runtime columns are intentionally absent.
ISETRACE_MAIN_METRICS = (
    "Coverage@256 Tokens",
    "Coverage@512 Tokens",
    "Coverage@1024 Tokens",
    "Coverage@2048 Tokens",
    "Coverage@4096 Tokens",
    "Coverage@8192 Tokens",
    "Full Support@256 Tokens",
    "Full Support@512 Tokens",
    "Full Support@1024 Tokens",
    "Full Support@2048 Tokens",
    "Full Support@4096 Tokens",
    "Full Support@8192 Tokens",
    "Coverage Budget-AUC",
    "Full Support Budget-AUC",
    "Recall@5",
    "MRR",
)

PER_TASK_METRICS = (
    "Coverage@256 Tokens",
    "Coverage@512 Tokens",
    "Coverage@1024 Tokens",
    "Coverage@2048 Tokens",
    "Coverage@4096 Tokens",
    "Coverage@8192 Tokens",
    "Full Support@256 Tokens",
    "Full Support@512 Tokens",
    "Full Support@1024 Tokens",
    "Full Support@2048 Tokens",
    "Full Support@4096 Tokens",
    "Full Support@8192 Tokens",
    "Coverage Budget-AUC",
    "Full Support Budget-AUC",
    "Span F1@2048 Tokens",
    "Evidence Density@2048 Tokens",
    "Recall@2",
    "Recall@5",
    "Recall@10",
    "Evidence F1@5",
    "Evidence F1@10",
    "Full Support@5",
    "Full Support@10",
    "MRR",
)


@dataclass(frozen=True)
class QueryMetadata:
    task_id: str
    graph_id: str
    memory_mode: str | None


@dataclass(frozen=True)
class RunSpec:
    label: str | None
    path: Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate fixed-test runs into a JSON main table with trainable "
            "mean/std, token-budget metrics, method-minus-baseline paired "
            "cluster-bootstrap intervals, and optional memory-mode strata."
        )
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help=(
            "Run output directory, optionally LABEL=PATH (repeatable). Explicit "
            "labels are required when one raw method has multiple supervision variants."
        ),
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Method label used as the paired-comparison baseline.",
    )
    parser.add_argument(
        "--trainable",
        action="append",
        default=[],
        help="Trainable method label (repeatable). Other labels are deterministic.",
    )
    parser.add_argument(
        "--delta-direction",
        choices=("method-minus-baseline", "baseline-minus-method"),
        default="method-minus-baseline",
        help=(
            "Paired delta direction. E2 relation controls use baseline-minus-method "
            "with full_rgcn as the baseline to report full minus control."
        ),
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        help=(
            "Optional JSONL sidecar used to attach trajectory clusters and memory "
            "modes to legacy per-task outputs. Supports query_id/trajectory_id or "
            "task_id/graph_id records."
        ),
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        help="Fail unless every run contains exactly this many unique tasks.",
    )
    parser.add_argument(
        "--task-ids",
        type=Path,
        help=(
            "Optional JSON array of task IDs. Filter every complete run to this exact "
            "subset before aggregation and fail if a selected task is absent."
        ),
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=PER_TASK_METRICS,
        default=[],
        help=(
            "Metric to aggregate and bootstrap (repeatable). By default all shared "
            "per-task metrics are analyzed."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional machine-readable main-table CSV (means plus separate std columns).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args(argv)

    metadata, metadata_source_digest = _load_query_metadata(args.query_metadata)
    selected_task_ids = _load_task_ids(args.task_ids)
    trainable_methods = set(args.trainable)
    rows = [
        _row_from_run(
            _parse_run_spec(value),
            trainable_methods=trainable_methods,
            query_metadata=metadata,
            selected_task_ids=selected_task_ids,
            expected_task_count=args.expected_task_count,
        )
        for value in args.runs
    ]
    if args.metrics:
        rows = [_select_metrics(row, args.metrics) for row in rows]
    result = analyze_main_results(
        rows,
        baseline_method=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        delta_direction=args.delta_direction.replace("-", "_"),
    )
    result["query_metadata_source_digest"] = metadata_source_digest
    result["query_metadata_subset_digest"] = _query_metadata_subset_digest(
        metadata,
        rows=rows,
    )
    if selected_task_ids is not None:
        result["task_subset"] = {
            "task_count": len(selected_task_ids),
            "task_ids_sha256": _sha256(args.task_ids),
        }
    write_json(args.output, result)
    if args.output_csv is not None:
        _write_main_table_csv(args.output_csv, result)
    return 0


def _row_from_run(
    spec: RunSpec,
    *,
    trainable_methods: set[str],
    query_metadata: Mapping[str, QueryMetadata],
    selected_task_ids: set[str] | None,
    expected_task_count: int | None,
) -> dict[str, object]:
    run = spec.path
    summary_path = run / "workflow" / "summary.yaml"
    per_task_path = run / "metrics" / "per_task.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    if not per_task_path.is_file():
        raise FileNotFoundError(f"missing per-task metrics: {per_task_path}")
    with summary_path.open(encoding="utf-8") as stream:
        summary = yaml.safe_load(stream)
    if not isinstance(summary, Mapping):
        raise ValueError(f"invalid workflow summary: {summary_path}")
    raw_method = summary.get("method")
    seed = summary.get("seed")
    if not isinstance(raw_method, str) or not raw_method:
        raise ValueError(f"summary has invalid method: {summary_path}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"summary has invalid seed: {summary_path}")
    dataset = summary.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"summary has invalid dataset: {summary_path}")
    variant = summary.get("variant")
    label = spec.label or (
        f"{raw_method}:{variant}"
        if isinstance(variant, str) and variant
        else raw_method
    )

    per_task: dict[str, dict[str, float]] = {}
    task_groups: dict[str, str] = {}
    task_strata: dict[str, str] = {}
    with per_task_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {per_task_path}:{line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"per-task row must be an object: {per_task_path}:{line_number}"
                )
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"missing task_id: {per_task_path}:{line_number}")
            if task_id in per_task:
                raise ValueError(f"duplicate task_id={task_id!r} in {per_task_path}")
            metrics = {
                metric: _numeric(
                    record[metric], path=f"{per_task_path}:{line_number}.{metric}"
                )
                for metric in PER_TASK_METRICS
                if metric in record and _is_number(record[metric])
            }
            if not metrics:
                raise ValueError(
                    f"per-task row has no supported metrics: {per_task_path}:{line_number}"
                )
            if dataset == "isetrace":
                missing_metrics = [
                    metric for metric in ISETRACE_MAIN_METRICS if metric not in metrics
                ]
                if missing_metrics:
                    raise ValueError(
                        f"ISETrace main-table metrics are missing from "
                        f"{per_task_path}:{line_number}: {missing_metrics}"
                    )
            per_task[task_id] = metrics
            sidecar = query_metadata.get(task_id)
            graph_id = _optional_string(record.get("graph_id"))
            memory_mode = _optional_string(record.get("memory_mode"))
            if sidecar is not None:
                if graph_id is not None and graph_id != sidecar.graph_id:
                    raise ValueError(f"task_id={task_id!r} has conflicting graph IDs")
                if memory_mode is not None and memory_mode != sidecar.memory_mode:
                    raise ValueError(
                        f"task_id={task_id!r} has conflicting memory modes"
                    )
                graph_id = sidecar.graph_id
                memory_mode = sidecar.memory_mode
            if graph_id is not None:
                task_groups[task_id] = graph_id
            if memory_mode is not None:
                task_strata[task_id] = memory_mode
    if selected_task_ids is not None:
        missing_selected = sorted(selected_task_ids - set(per_task))
        if missing_selected:
            raise ValueError(
                f"run={run} is missing selected tasks: {missing_selected[:5]}"
            )
        per_task = {task_id: per_task[task_id] for task_id in selected_task_ids}
        task_groups = {
            task_id: task_groups[task_id]
            for task_id in selected_task_ids
            if task_id in task_groups
        }
        task_strata = {
            task_id: task_strata[task_id]
            for task_id in selected_task_ids
            if task_id in task_strata
        }
    if not per_task:
        raise ValueError(f"no per-task rows in {per_task_path}")
    if expected_task_count is not None and len(per_task) != expected_task_count:
        raise ValueError(
            f"run={run} expected {expected_task_count} tasks, observed {len(per_task)}"
        )
    if query_metadata:
        missing = sorted(set(per_task) - set(query_metadata))
        if missing:
            raise ValueError(
                f"query metadata does not cover run={run}; missing={missing[:5]}"
            )
    if task_groups and set(task_groups) != set(per_task):
        raise ValueError(f"trajectory cluster IDs are incomplete for run={run}")
    if task_strata and set(task_strata) != set(per_task):
        raise ValueError(f"memory-mode strata are incomplete for run={run}")

    metrics = _aggregate_from_per_task(per_task)
    return {
        "method": label,
        "trainable": label in trainable_methods,
        "seed": seed,
        "metrics": metrics,
        "per_task": per_task,
        "task_groups": task_groups,
        "task_strata": task_strata,
        "test_artifact_digest": _test_artifact_digest(run),
    }


def _select_metrics(
    row: dict[str, object], selected_metrics: Sequence[str]
) -> dict[str, object]:
    metrics = cast(Mapping[str, float], row["metrics"])
    per_task = cast(Mapping[str, Mapping[str, float]], row["per_task"])
    missing_aggregate = [
        metric for metric in selected_metrics if metric not in metrics
    ]
    if missing_aggregate:
        raise ValueError(f"run lacks selected aggregate metrics: {missing_aggregate}")
    selected_per_task: dict[str, dict[str, float]] = {}
    for task_id, task_metrics in per_task.items():
        missing = [metric for metric in selected_metrics if metric not in task_metrics]
        if missing:
            raise ValueError(
                f"task_id={task_id!r} lacks selected per-task metrics: {missing}"
            )
        selected_per_task[task_id] = {
            metric: task_metrics[metric] for metric in selected_metrics
        }
    selected = dict(row)
    selected["metrics"] = {metric: metrics[metric] for metric in selected_metrics}
    selected["per_task"] = selected_per_task
    return selected


def _aggregate_from_per_task(
    per_task: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    task_count = len(per_task)
    for metric in PER_TASK_METRICS:
        values = [
            task_metrics[metric]
            for task_metrics in per_task.values()
            if metric in task_metrics
        ]
        if len(values) == task_count and task_count > 0:
            metrics[metric] = sum(values) / task_count
    return metrics


def _parse_run_spec(value: str) -> RunSpec:
    direct = Path(value)
    if direct.exists() or "=" not in value:
        return RunSpec(label=None, path=direct)
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"invalid --run value={value!r}; expected LABEL=PATH")
    return RunSpec(label=label, path=Path(raw_path))


def _load_task_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"task-ID file must be a nonempty JSON array: {path}")
    task_ids: set[str] = set()
    for index, task_id in enumerate(value):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"task-ID file has invalid item at index={index}: {path}")
        if task_id in task_ids:
            raise ValueError(f"task-ID file has duplicate task_id={task_id!r}: {path}")
        task_ids.add(task_id)
    return task_ids


def _load_query_metadata(
    path: Path | None,
) -> tuple[dict[str, QueryMetadata], str | None]:
    if path is None:
        return {}, None
    metadata: dict[str, QueryMetadata] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"metadata row must be an object: {path}:{line_number}"
                )
            task_id = record.get("task_id", record.get("query_id"))
            graph_id = record.get("graph_id", record.get("trajectory_id"))
            memory_mode = record.get("memory_mode")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(
                    f"metadata row lacks task/query ID: {path}:{line_number}"
                )
            if not isinstance(graph_id, str) or not graph_id:
                raise ValueError(
                    f"metadata row lacks graph/trajectory ID: {path}:{line_number}"
                )
            if memory_mode is not None and not isinstance(memory_mode, str):
                raise ValueError(
                    f"metadata row has invalid memory_mode: {path}:{line_number}"
                )
            value = QueryMetadata(task_id, graph_id, memory_mode)
            previous = metadata.get(task_id)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting metadata for task_id={task_id!r}")
            if previous is not None:
                raise ValueError(f"duplicate metadata task_id={task_id!r}")
            metadata[task_id] = value
    if not metadata:
        raise ValueError(f"query metadata file is empty: {path}")
    return metadata, _sha256(path)


def _query_metadata_subset_digest(
    metadata: Mapping[str, QueryMetadata],
    *,
    rows: Sequence[Mapping[str, object]],
) -> str | None:
    if not metadata or not rows:
        return None
    per_task = rows[0].get("per_task")
    if not isinstance(per_task, Mapping):
        raise ValueError("analysis row has no per_task mapping")
    task_ids = sorted(task_id for task_id in per_task if isinstance(task_id, str))
    payload = "".join(
        f"{task_id}\0{metadata[task_id].graph_id}\0"
        f"{metadata[task_id].memory_mode or ''}\n"
        for task_id in task_ids
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _test_artifact_digest(run: Path) -> str | None:
    path = run / "assets" / "manifest.yaml"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"invalid assets manifest: {path}")
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"invalid assets manifest: {path}")
    candidates: list[str] = []
    for asset in assets:
        if not isinstance(asset, Mapping) or asset.get("kind") != "dataset":
            continue
        origin = asset.get("origin")
        metadata = asset.get("metadata")
        split = None
        if isinstance(origin, Mapping):
            split = origin.get("split")
        if split is None and isinstance(metadata, Mapping):
            split = metadata.get("split")
        digest = asset.get("digest")
        if split == "test" and isinstance(digest, str):
            candidates.append(digest)
    if len(candidates) > 1:
        raise ValueError(f"assets manifest has multiple test datasets: {path}")
    return candidates[0] if candidates else None


def _write_main_table_csv(path: Path, result: Mapping[str, object]) -> None:
    raw_summary_value = result.get("summary")
    if not isinstance(raw_summary_value, Mapping):
        raise ValueError("analysis result has no summary mapping")
    raw_summary = cast(Mapping[object, object], raw_summary_value)
    metric_names: list[str] = []
    for method_value in raw_summary.values():
        if not isinstance(method_value, Mapping):
            continue
        metrics_value = method_value.get("metrics")
        if not isinstance(metrics_value, Mapping):
            continue
        metric_names.extend(
            metric for metric in metrics_value if isinstance(metric, str)
        )
    metric_names = sorted(set(metric_names))
    fields = ["Method", "Trainable", "Seeds"]
    for metric in metric_names:
        fields.extend((metric, f"{metric} Std"))
    rows: list[dict[str, object]] = []
    for label, method_value in raw_summary.items():
        if not isinstance(label, str) or not isinstance(method_value, Mapping):
            continue
        method = cast(Mapping[object, object], method_value)
        seeds_value = method.get("seeds")
        seeds = seeds_value if isinstance(seeds_value, list) else []
        row: dict[str, object] = {
            "Method": label,
            "Trainable": method.get("trainable"),
            "Seeds": ",".join(str(seed) for seed in seeds),
        }
        metrics = method.get("metrics")
        if isinstance(metrics, Mapping):
            for metric, values in metrics.items():
                if not isinstance(metric, str) or not isinstance(values, Mapping):
                    continue
                row[metric] = values.get("mean", values.get("value"))
                row[f"{metric} Std"] = values.get("std", "")
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _numeric(value: object, *, path: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


if __name__ == "__main__":
    raise SystemExit(main())
