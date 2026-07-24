from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.analysis.paired_bootstrap import paired_delta_bootstrap_ci


@dataclass(frozen=True)
class _NormalizedRow:
    method: str
    trainable: bool
    seed: int
    metrics: dict[str, float]
    per_task: dict[str, dict[str, float]]


def analyze_main_results(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_method: str,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 13,
) -> dict[str, object]:
    """Aggregate main-results rows into per-method summaries and paired CIs.

    Each row describes one evaluated run: its ``method``, whether the method is
    ``trainable``, the run ``seed``, aggregate ``metrics``, and ``per_task``
    metrics keyed by ``task_id``. Trainable methods contribute multiple seed
    rows; deterministic methods contribute exactly one.

    Returns per-method mean/std (trainable) or a single value (deterministic),
    plus a per-query paired bootstrap 95% CI on the baseline-minus-method delta
    for every method compared against ``baseline_method``.

    Every compared row must have evaluated the identical set of test task ids;
    a mismatch raises rather than silently intersecting.
    """
    normalized = [_normalize_row(row) for row in rows]
    if not normalized:
        raise ValueError("Main-results analysis requires at least one row.")

    by_method_seed = {(row.method, row.seed): row for row in normalized}
    if len(by_method_seed) != len(normalized):
        raise ValueError("Main-results analysis requires unique (method, seed) rows.")

    methods = sorted({row.method for row in normalized})
    if baseline_method not in methods:
        raise ValueError(f"Missing baseline method={baseline_method!r}.")

    # Every compared per-task population must share the same test task ids.
    task_id_reference = _shared_task_ids(normalized)

    metric_names = sorted(set.intersection(*(set(row.metrics) for row in normalized)))
    per_task_metric_names = sorted(
        set.intersection(
            *(
                {
                    metric
                    for task_metrics in row.per_task.values()
                    for metric in task_metrics
                }
                for row in normalized
            )
        )
    )

    _enforce_trainable_consistency(normalized)

    summary: dict[str, object] = {}
    for method in methods:
        method_rows = [row for row in normalized if row.method == method]
        trainable = method_rows[0].trainable
        seeds = sorted(row.seed for row in method_rows)
        metric_summary: dict[str, object] = {}
        for metric in metric_names:
            values = [row.metrics[metric] for row in method_rows]
            if trainable:
                metric_summary[metric] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "seed_values": values,
                }
            else:
                metric_summary[metric] = {"value": values[0]}
        summary[method] = {
            "trainable": trainable,
            "seeds": seeds,
            "metrics": metric_summary,
        }

    baseline_rows = [row for row in normalized if row.method == baseline_method]
    paired: dict[str, object] = {}
    for method in methods:
        if method == baseline_method:
            continue
        method_rows = [row for row in normalized if row.method == method]
        metric_results: dict[str, object] = {}
        for metric in per_task_metric_names:
            deltas = _paired_deltas(
                baseline_rows=baseline_rows,
                method_rows=method_rows,
                metric=metric,
            )
            metric_results[metric] = paired_delta_bootstrap_ci(
                deltas,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        paired[method] = {"metrics": metric_results}

    return {
        "baseline_method": baseline_method,
        "test_task_count": len(task_id_reference),
        "summary": summary,
        "paired_analysis": paired,
    }


def _paired_deltas(
    *,
    baseline_rows: Sequence[_NormalizedRow],
    method_rows: Sequence[_NormalizedRow],
    metric: str,
) -> list[float]:
    """Per-query baseline-minus-method deltas aligned by task id.

    When the baseline is deterministic (single seed), every method seed is
    paired against that single baseline population. When both sides have
    matching seeds, they are paired seed-by-seed. Otherwise the baseline is
    paired against each method seed using its single available population.
    """
    baseline_by_seed = {row.seed: row for row in baseline_rows}
    deltas: list[float] = []
    for method_row in method_rows:
        if method_row.seed in baseline_by_seed:
            baseline_row = baseline_by_seed[method_row.seed]
        elif len(baseline_rows) == 1:
            baseline_row = baseline_rows[0]
        else:
            raise ValueError(
                f"No baseline seed to pair with method={method_row.method!r} "
                f"seed={method_row.seed}."
            )
        for task_id in sorted(baseline_row.per_task):
            baseline_value = baseline_row.per_task[task_id].get(metric)
            method_value = method_row.per_task[task_id].get(metric)
            if baseline_value is None or method_value is None:
                continue
            deltas.append(baseline_value - method_value)
    return deltas


def _shared_task_ids(rows: Sequence[_NormalizedRow]) -> set[str]:
    reference: set[str] | None = None
    reference_label = ""
    for row in rows:
        task_ids = set(row.per_task)
        if reference is None:
            reference = task_ids
            reference_label = f"{row.method}@seed{row.seed}"
            continue
        if task_ids != reference:
            missing = sorted(reference - task_ids)
            extra = sorted(task_ids - reference)
            raise ValueError(
                "Compared methods must share the same test split. "
                f"{row.method}@seed{row.seed} differs from {reference_label}: "
                f"missing={missing[:5]} extra={extra[:5]}."
            )
    return reference or set()


def _enforce_trainable_consistency(rows: Sequence[_NormalizedRow]) -> None:
    by_method: dict[str, set[bool]] = {}
    for row in rows:
        by_method.setdefault(row.method, set()).add(row.trainable)
    for method, flags in by_method.items():
        if len(flags) > 1:
            raise ValueError(
                f"method={method!r} has inconsistent trainable flags across rows."
            )


def _normalize_row(row: Mapping[str, object]) -> _NormalizedRow:
    method = row.get("method")
    trainable = row.get("trainable")
    seed = row.get("seed")
    metrics = row.get("metrics")
    per_task = row.get("per_task")
    if not isinstance(method, str) or not method:
        raise ValueError("Main-results row method must be a non-empty string.")
    if not isinstance(trainable, bool):
        raise ValueError("Main-results row trainable must be a boolean.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Main-results row seed must be an integer.")
    if not isinstance(metrics, Mapping) or not isinstance(per_task, Mapping):
        raise ValueError("Main-results rows require metrics and per_task mappings.")
    normalized_metrics = {
        str(name): _number(value, f"metrics.{name}") for name, value in metrics.items()
    }
    normalized_tasks: dict[str, dict[str, float]] = {}
    for task_id, raw_task_metrics in per_task.items():
        if not isinstance(task_id, str) or not isinstance(raw_task_metrics, Mapping):
            raise ValueError("per_task must map task ids to metric mappings.")
        normalized_tasks[task_id] = {
            str(name): _number(value, f"per_task.{task_id}.{name}")
            for name, value in raw_task_metrics.items()
        }
    return _NormalizedRow(
        method=method,
        trainable=trainable,
        seed=seed,
        metrics=normalized_metrics,
        per_task=normalized_tasks,
    )


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric.")
    return float(value)


__all__ = ["analyze_main_results"]
