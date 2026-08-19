from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.io import write_json
from scripts.aggregate_main_results import main as aggregate_main_results

TRAINING_SIZES = ((10, 241), (25, 603), (50, 1205), (100, 2410))
MODEL_SEEDS = (13, 17, 29, 37, 41)
DEV_TRAJECTORIES = 352
TEST_TRAJECTORIES = 1207
TEST_TASKS = 2000
SPLIT_SEED = 13
METRICS = ("Coverage@1024 Tokens", "Full Support@2048 Tokens")
METHODS = {
    "dense_ft_provenance_unit": ("dft_pu", "dense_ft", "provenance_unit"),
    "exact_seed_passthrough": ("rgcn_nograph", "provenance_rgcn", "wo_graph"),
    "residual_rgcn": ("rgcn_full", "provenance_rgcn", "full_rgcn"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the frozen ISETrace Training-Size Robustness cohort over "
            "five seeds and trajectory-cluster bootstrap intervals."
        )
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=Path(
            "data/isetrace/query-authoring/isetrace-v7-raw.jsonl.metadata.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args(argv)

    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    run_root = args.run_root.resolve()
    query_metadata = args.query_metadata.resolve()
    if not query_metadata.is_file():
        raise FileNotFoundError(f"missing query metadata: {query_metadata}")

    points: list[dict[str, object]] = []
    runs: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="isetrace-training-size-") as temp_dir:
        temporary_root = Path(temp_dir)
        for percentage, train_trajectories in TRAINING_SIZES:
            cohort = _cohort_runs(
                run_root,
                percentage=percentage,
                train_trajectories=train_trajectories,
            )
            runs[str(percentage)] = [
                str(path.relative_to(run_root))
                for label in METHODS
                for path in cohort[label]
            ]
            aggregate_path = temporary_root / f"training-size-{percentage}.json"
            aggregate_args = [
                "--baseline",
                "exact_seed_passthrough",
                "--query-metadata",
                str(query_metadata),
                "--expected-task-count",
                str(TEST_TASKS),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
                "--output",
                str(aggregate_path),
            ]
            for label, paths in cohort.items():
                aggregate_args.extend(("--trainable", label))
                for path in paths:
                    aggregate_args.extend(("--run", f"{label}={path}"))
            if aggregate_main_results(aggregate_args) != 0:
                raise RuntimeError(
                    f"aggregation failed for training percentage={percentage}"
                )
            aggregate = _read_mapping(aggregate_path)
            points.append(
                _training_size_point(
                    aggregate,
                    percentage=percentage,
                    train_trajectories=train_trajectories,
                )
            )

    _validate_shared_test(points)
    result: dict[str, object] = {
        "schema_version": 1,
        "study": "isetrace_training_size_robustness",
        "dataset": "isetrace",
        "split_seed": SPLIT_SEED,
        "full_train_trajectory_count": TRAINING_SIZES[-1][1],
        "dev_trajectory_count": DEV_TRAJECTORIES,
        "test_trajectory_count": TEST_TRAJECTORIES,
        "test_task_count": TEST_TASKS,
        "seeds": list(MODEL_SEEDS),
        "metrics": list(METRICS),
        "bootstrap": {
            "unit": "trajectory",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "seed_pair_reduction": "mean per task before cluster resampling",
        },
        "fixed_recipe": {
            "dense_ft_epochs": 1,
            "residual_rgcn_epochs": 15,
            "per_size_hyperparameter_tuning": False,
        },
        "points": points,
        "runs": runs,
    }
    write_json(args.output, result)
    _write_csv(args.output_csv, points)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_render_report(result), encoding="utf-8", newline="\n")
    if args.figure is not None:
        _write_figure(args.figure, points)
    return 0


def _cohort_runs(
    run_root: Path,
    *,
    percentage: int,
    train_trajectories: int,
) -> dict[str, tuple[Path, ...]]:
    cohort: dict[str, tuple[Path, ...]] = {}
    for label, (name_token, method, variant) in METHODS.items():
        paths = tuple(
            run_root / f"isetrace_v7_trainsize{percentage}_{name_token}_s{seed}"
            for seed in MODEL_SEEDS
        )
        for seed, path in zip(MODEL_SEEDS, paths, strict=True):
            _validate_run(
                path,
                method=method,
                variant=variant,
                seed=seed,
                train_trajectories=train_trajectories,
            )
        cohort[label] = paths
    return cohort


def _validate_run(
    path: Path,
    *,
    method: str,
    variant: str,
    seed: int,
    train_trajectories: int,
) -> None:
    summary = _read_mapping(path / "workflow" / "summary.yaml")
    expected_summary = {
        "method": method,
        "variant": variant,
        "dataset": "isetrace",
        "profile": "full",
        "seed": seed,
    }
    observed_summary = {key: summary.get(key) for key in expected_summary}
    if observed_summary != expected_summary:
        raise ValueError(
            f"run summary mismatch: path={path} "
            f"expected={expected_summary} observed={observed_summary}"
        )

    config = _read_mapping(path / "config" / "resolved.yaml")
    dataset = _mapping(config.get("dataset"), path=f"{path}.dataset")
    trajectories = _mapping(
        dataset.get("trajectories"), path=f"{path}.dataset.trajectories"
    )
    splits = _mapping(
        trajectories.get("splits"), path=f"{path}.dataset.trajectories.splits"
    )
    expected_splits = {
        "train": train_trajectories,
        "dev": DEV_TRAJECTORIES,
        "test": TEST_TRAJECTORIES,
    }
    observed_splits = {key: splits.get(key) for key in expected_splits}
    if observed_splits != expected_splits:
        raise ValueError(
            f"trajectory split mismatch: path={path} "
            f"expected={expected_splits} observed={observed_splits}"
        )
    if config.get("split_seed") != SPLIT_SEED:
        raise ValueError(
            f"run={path} requires split_seed={SPLIT_SEED}, "
            f"observed={config.get('split_seed')!r}"
        )


def _training_size_point(
    aggregate: Mapping[str, object],
    *,
    percentage: int,
    train_trajectories: int,
) -> dict[str, object]:
    summary = _mapping(aggregate.get("summary"), path="summary")
    paired = _mapping(aggregate.get("paired_analysis"), path="paired_analysis")
    methods: dict[str, object] = {}
    effects: dict[str, object] = {}
    for label in METHODS:
        method = _mapping(summary.get(label), path=f"summary.{label}")
        metric_values = _mapping(
            method.get("metrics"), path=f"summary.{label}.metrics"
        )
        methods[label] = {
            "seeds": method.get("seeds"),
            "metrics": {metric: metric_values[metric] for metric in METRICS},
        }
        if label == "exact_seed_passthrough":
            effects[label] = {
                "metrics": {
                    metric: {
                        "mean_delta": 0.0,
                        "ci_95": [0.0, 0.0],
                        "seed_pair_count": len(MODEL_SEEDS),
                    }
                    for metric in METRICS
                }
            }
            continue
        comparison = _mapping(paired.get(label), path=f"paired_analysis.{label}")
        comparison_metrics = _mapping(
            comparison.get("metrics"), path=f"paired_analysis.{label}.metrics"
        )
        effects[label] = {
            "metrics": {metric: comparison_metrics[metric] for metric in METRICS}
        }
    return {
        "percentage": percentage,
        "train_trajectory_count": train_trajectories,
        "test_artifact_digest": aggregate.get("test_artifact_digest"),
        "test_task_count": aggregate.get("test_task_count"),
        "test_cluster_count": aggregate.get("test_cluster_count"),
        "methods": methods,
        "effects_vs_exact_seed": effects,
    }


def _validate_shared_test(points: Sequence[Mapping[str, object]]) -> None:
    digests = {point.get("test_artifact_digest") for point in points}
    task_counts = {point.get("test_task_count") for point in points}
    cluster_counts = {point.get("test_cluster_count") for point in points}
    if len(digests) != 1 or None in digests:
        raise ValueError(f"training sizes use inconsistent test artifacts: {digests}")
    if task_counts != {TEST_TASKS}:
        raise ValueError(f"training sizes use inconsistent test tasks: {task_counts}")
    if cluster_counts != {TEST_TRAJECTORIES}:
        raise ValueError(
            f"training sizes use inconsistent test trajectories: {cluster_counts}"
        )


def _write_csv(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for point in points:
        methods = _mapping(point.get("methods"), path="point.methods")
        effects = _mapping(
            point.get("effects_vs_exact_seed"), path="point.effects_vs_exact_seed"
        )
        for label in METHODS:
            method = _mapping(methods.get(label), path=f"point.methods.{label}")
            metrics = _mapping(
                method.get("metrics"), path=f"point.methods.{label}.metrics"
            )
            effect_metrics = _mapping(
                _mapping(
                    effects.get(label), path=f"point.effects_vs_exact_seed.{label}"
                ).get("metrics"),
                path=f"point.effects_vs_exact_seed.{label}.metrics",
            )
            for metric in METRICS:
                values = _mapping(metrics.get(metric), path=f"metrics.{metric}")
                effect = _mapping(
                    effect_metrics.get(metric), path=f"effects.{metric}"
                )
                ci = cast(Sequence[object], effect["ci_95"])
                rows.append(
                    {
                        "percentage": point["percentage"],
                        "train_trajectory_count": point["train_trajectory_count"],
                        "method": label,
                        "metric": metric,
                        "mean": values["mean"],
                        "std": values["std"],
                        "mean_delta_vs_exact_seed": effect["mean_delta"],
                        "ci_95_low": ci[0],
                        "ci_95_high": ci[1],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(result: Mapping[str, object]) -> str:
    lines = [
        "# ISETrace Training-Size Robustness",
        "",
        "## Protocol",
        "",
        "- Train trajectories: 241, 603, 1,205, and 2,410 nested prefixes.",
        "- Fixed dev/test trajectories: 352 / 1,207.",
        "- Seeds: 13, 17, 29, 37, and 41.",
        "- Intervals: paired trajectory-cluster bootstrap after seed averaging.",
        "- Fixed recipes: PU Dense-FT 1 epoch; residual R-GCN 15 epochs.",
        "",
    ]
    points = cast(Sequence[Mapping[str, object]], result["points"])
    for metric in METRICS:
        lines.extend((f"## {metric}", ""))
        lines.append(
            "| Train | PU Dense-FT | Exact seed | Residual R-GCN | Residual gain |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for point in points:
            methods = _mapping(point["methods"], path="point.methods")
            effects = _mapping(
                point["effects_vs_exact_seed"], path="point.effects_vs_exact_seed"
            )
            values = [
                _summary_cell(
                    _mapping(
                        _mapping(methods[label], path=label)["metrics"],
                        path=f"{label}.metrics",
                    )[metric]
                )
                for label in METHODS
            ]
            residual_effects = _mapping(
                _mapping(effects["residual_rgcn"], path="residual_rgcn")["metrics"],
                path="residual_rgcn.metrics",
            )
            effect = _effect_cell(
                _mapping(residual_effects[metric], path=f"residual.{metric}")
            )
            lines.append(
                f"| {point['percentage']}% ({point['train_trajectory_count']}) "
                f"| {values[0]} | {values[1]} | {values[2]} | {effect} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _summary_cell(value: object) -> str:
    metric = _mapping(value, path="summary metric")
    return f"{100.0 * _number(metric['mean']):.2f} +/- {100.0 * _number(metric['std']):.2f}"


def _effect_cell(value: Mapping[str, object]) -> str:
    ci = cast(Sequence[object], value["ci_95"])
    return (
        f"{100.0 * _number(value['mean_delta']):+.2f} pp "
        f"[{100.0 * _number(ci[0]):+.2f}, {100.0 * _number(ci[1]):+.2f}]"
    )


def _write_figure(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "matplotlib is required to render the training-size figure"
        ) from error

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    styles = {
        "dense_ft_provenance_unit": ("#009E73", "^", "PU Dense-FT"),
        "exact_seed_passthrough": ("#0072B2", "o", "Exact seed"),
        "residual_rgcn": ("#D55E00", "s", "Residual R-GCN"),
    }
    x_values = [int(_number(point["percentage"])) for point in points]
    for axis, metric in zip(axes, METRICS, strict=True):
        for label, (color, marker, legend_label) in styles.items():
            means: list[float] = []
            stds: list[float] = []
            for point in points:
                methods = _mapping(point["methods"], path="point.methods")
                method = _mapping(methods[label], path=label)
                metrics = _mapping(method["metrics"], path=f"{label}.metrics")
                values = _mapping(metrics[metric], path=f"{label}.{metric}")
                means.append(100.0 * _number(values["mean"]))
                stds.append(100.0 * _number(values["std"]))
            axis.errorbar(
                x_values,
                means,
                yerr=stds,
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.35,
                capsize=2.5,
                label=legend_label,
            )
        axis.set_xticks(x_values, [f"{value}%" for value in x_values])
        axis.set_xlabel("Training trajectories (% of 2,410)", fontsize=8.5)
        axis.set_title(metric.replace(" Tokens", ""), fontsize=9.2)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.tick_params(labelsize=7.8)
    axes[0].set_ylabel("Mean test score (%) +/- seed SD", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _read_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    return _mapping(value, path=str(path))


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def _number(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"expected numeric value, observed={value!r}")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
