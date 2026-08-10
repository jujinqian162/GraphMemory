from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.analysis.paired_bootstrap import (
    paired_cluster_delta_bootstrap_ci,
)


@dataclass(frozen=True)
class _NormalizedRow:
    method: str
    trainable: bool
    seed: int
    metrics: dict[str, float]
    per_task: dict[str, dict[str, float]]
    task_groups: dict[str, str]
    task_strata: dict[str, str]
    test_artifact_digest: str | None


def analyze_main_results(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_method: str,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 13,
) -> dict[str, object]:
    """Aggregate fixed-test runs and compute paired cluster-bootstrap CIs.

    Trainable methods are summarized as mean and sample standard deviation over
    model-training seeds. Deterministic methods must contribute one row. Paired
    deltas use ``method - baseline`` and are first averaged over matched seed
    pairs per task, then bootstrapped by ``task_groups``. ISETrace callers should
    map every task to its source trajectory; generic callers may omit groups and
    will fall back to one cluster per task.
    """
    normalized = [_normalize_row(row) for row in rows]
    if not normalized:
        raise ValueError("Main-results analysis requires at least one row.")

    by_method_seed = {(row.method, row.seed): row for row in normalized}
    if len(by_method_seed) != len(normalized):
        raise ValueError(
            "Main-results analysis requires unique (method label, seed) rows. "
            "Use distinct labels for supervision variants."
        )

    methods = sorted({row.method for row in normalized})
    if baseline_method not in methods:
        raise ValueError(f"Missing baseline method={baseline_method!r}.")

    task_ids = _shared_task_ids(normalized)
    task_groups, task_strata = _shared_task_metadata(normalized, task_ids)
    test_artifact_digest = _shared_test_artifact_digest(normalized)
    metric_names = _shared_aggregate_metrics(normalized)
    per_task_metric_names = _shared_per_task_metrics(normalized)
    _enforce_method_consistency(normalized)

    summary = _method_summary(normalized, metric_names=metric_names)
    stratified_summary = _stratified_method_summary(
        normalized,
        metric_names=per_task_metric_names,
        task_strata=task_strata,
    )

    baseline_rows = [row for row in normalized if row.method == baseline_method]
    paired: dict[str, object] = {}
    stratified_paired: dict[str, object] = {}
    strata = sorted(set(task_strata.values()))
    for method in methods:
        if method == baseline_method:
            continue
        method_rows = [row for row in normalized if row.method == method]
        metric_results: dict[str, object] = {}
        stratum_results: dict[str, dict[str, dict[str, object]]] = {
            stratum: {"metrics": {}} for stratum in strata
        }
        for metric in per_task_metric_names:
            deltas, seed_pair_count = _paired_task_deltas(
                baseline_rows=baseline_rows,
                method_rows=method_rows,
                metric=metric,
            )
            result = _cluster_bootstrap(
                deltas,
                task_groups=task_groups,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            result.update(
                {
                    "delta_direction": "method_minus_baseline",
                    "seed_pair_count": seed_pair_count,
                }
            )
            metric_results[metric] = result
            for stratum in strata:
                stratum_deltas = {
                    task_id: delta
                    for task_id, delta in deltas.items()
                    if task_strata.get(task_id) == stratum
                }
                stratum_result = _cluster_bootstrap(
                    stratum_deltas,
                    task_groups=task_groups,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                stratum_result.update(
                    {
                        "delta_direction": "method_minus_baseline",
                        "seed_pair_count": seed_pair_count,
                    }
                )
                stratum_results[stratum]["metrics"][metric] = stratum_result
        paired[method] = {"metrics": metric_results}
        if stratum_results:
            stratified_paired[method] = stratum_results

    cluster_unit = (
        "task"
        if all(task_groups[task_id] == task_id for task_id in task_ids)
        else "group"
    )
    return {
        "baseline_method": baseline_method,
        "delta_direction": "method_minus_baseline",
        "test_task_count": len(task_ids),
        "test_cluster_count": len(set(task_groups.values())),
        "cluster_unit": cluster_unit,
        "test_artifact_digest": test_artifact_digest,
        "summary": summary,
        "stratified_summary": stratified_summary,
        "paired_analysis": paired,
        "stratified_paired_analysis": stratified_paired,
    }


def _method_summary(
    rows: Sequence[_NormalizedRow],
    *,
    metric_names: Sequence[str],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for method in sorted({row.method for row in rows}):
        method_rows = sorted(
            (row for row in rows if row.method == method), key=lambda row: row.seed
        )
        trainable = method_rows[0].trainable
        metric_summary = {
            metric: _summarize_values(
                [row.metrics[metric] for row in method_rows], trainable=trainable
            )
            for metric in metric_names
        }
        summary[method] = {
            "trainable": trainable,
            "seeds": [row.seed for row in method_rows],
            "metrics": metric_summary,
        }
    return summary


def _stratified_method_summary(
    rows: Sequence[_NormalizedRow],
    *,
    metric_names: Sequence[str],
    task_strata: Mapping[str, str],
) -> dict[str, object]:
    if not task_strata:
        return {}
    result: dict[str, object] = {}
    for method in sorted({row.method for row in rows}):
        method_rows = sorted(
            (row for row in rows if row.method == method), key=lambda row: row.seed
        )
        trainable = method_rows[0].trainable
        by_stratum: dict[str, object] = {}
        for stratum in sorted(set(task_strata.values())):
            stratum_task_ids = sorted(
                task_id
                for task_id, task_stratum in task_strata.items()
                if task_stratum == stratum
            )
            metrics: dict[str, object] = {}
            for metric in metric_names:
                seed_values = [
                    statistics.fmean(
                        row.per_task[task_id][metric] for task_id in stratum_task_ids
                    )
                    for row in method_rows
                ]
                metrics[metric] = _summarize_values(seed_values, trainable=trainable)
            by_stratum[stratum] = {
                "task_count": len(stratum_task_ids),
                "metrics": metrics,
            }
        result[method] = by_stratum
    return result


def _summarize_values(values: Sequence[float], *, trainable: bool) -> dict[str, object]:
    if trainable:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seed_values": list(values),
        }
    return {"value": values[0]}


def _paired_task_deltas(
    *,
    baseline_rows: Sequence[_NormalizedRow],
    method_rows: Sequence[_NormalizedRow],
    metric: str,
) -> tuple[dict[str, float], int]:
    pairs = _seed_pairs(baseline_rows, method_rows)
    task_ids = sorted(pairs[0][0].per_task)
    return (
        {
            task_id: statistics.fmean(
                method_row.per_task[task_id][metric]
                - baseline_row.per_task[task_id][metric]
                for baseline_row, method_row in pairs
            )
            for task_id in task_ids
        },
        len(pairs),
    )


def _seed_pairs(
    baseline_rows: Sequence[_NormalizedRow],
    method_rows: Sequence[_NormalizedRow],
) -> list[tuple[_NormalizedRow, _NormalizedRow]]:
    baseline_by_seed = {row.seed: row for row in baseline_rows}
    method_by_seed = {row.seed: row for row in method_rows}
    if len(baseline_rows) == 1 and len(method_rows) == 1:
        return [(baseline_rows[0], method_rows[0])]
    if set(baseline_by_seed) == set(method_by_seed):
        return [
            (baseline_by_seed[seed], method_by_seed[seed])
            for seed in sorted(baseline_by_seed)
        ]
    if len(baseline_rows) == 1:
        return [
            (baseline_rows[0], row)
            for row in sorted(method_rows, key=lambda row: row.seed)
        ]
    if len(method_rows) == 1:
        return [
            (row, method_rows[0])
            for row in sorted(baseline_rows, key=lambda row: row.seed)
        ]
    raise ValueError(
        "Trainable baseline and method require identical seed sets for paired analysis."
    )


def _cluster_bootstrap(
    deltas: Mapping[str, float],
    *,
    task_groups: Mapping[str, str],
    samples: int,
    seed: int,
) -> dict[str, object]:
    by_cluster: dict[str, list[float]] = {}
    for task_id, delta in deltas.items():
        by_cluster.setdefault(task_groups[task_id], []).append(delta)
    return paired_cluster_delta_bootstrap_ci(
        by_cluster,
        samples=samples,
        seed=seed,
    )


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


def _shared_task_metadata(
    rows: Sequence[_NormalizedRow],
    task_ids: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    groups: dict[str, str] = {}
    strata: dict[str, str] = {}
    for task_id in task_ids:
        observed_groups = {row.task_groups[task_id] for row in rows}
        if len(observed_groups) != 1:
            raise ValueError(f"task_id={task_id!r} has inconsistent cluster IDs.")
        groups[task_id] = next(iter(observed_groups))
        observed_strata = {
            row.task_strata[task_id] for row in rows if task_id in row.task_strata
        }
        if observed_strata:
            if len(observed_strata) != 1 or any(
                task_id not in row.task_strata for row in rows
            ):
                raise ValueError(f"task_id={task_id!r} has inconsistent strata.")
            strata[task_id] = next(iter(observed_strata))
    return groups, strata


def _shared_test_artifact_digest(rows: Sequence[_NormalizedRow]) -> str | None:
    observed = {row.test_artifact_digest for row in rows}
    if len(observed) > 1:
        raise ValueError(
            "Compared runs reference different or missing test artifact digests: "
            f"{sorted(str(value) for value in observed)}"
        )
    return next(iter(observed))


def _shared_aggregate_metrics(rows: Sequence[_NormalizedRow]) -> list[str]:
    return sorted(set.intersection(*(set(row.metrics) for row in rows)))


def _shared_per_task_metrics(rows: Sequence[_NormalizedRow]) -> list[str]:
    per_row: list[set[str]] = []
    for row in rows:
        task_metric_sets = [set(metrics) for metrics in row.per_task.values()]
        per_row.append(set.intersection(*task_metric_sets))
    return sorted(set.intersection(*per_row))


def _enforce_method_consistency(rows: Sequence[_NormalizedRow]) -> None:
    by_method: dict[str, list[_NormalizedRow]] = {}
    for row in rows:
        by_method.setdefault(row.method, []).append(row)
    for method, method_rows in by_method.items():
        flags = {row.trainable for row in method_rows}
        if len(flags) > 1:
            raise ValueError(
                f"method={method!r} has inconsistent trainable flags across rows."
            )
        if not method_rows[0].trainable and len(method_rows) != 1:
            raise ValueError(
                f"deterministic method={method!r} must contribute exactly one run."
            )


def _normalize_row(row: Mapping[str, object]) -> _NormalizedRow:
    method = row.get("method")
    trainable = row.get("trainable")
    seed = row.get("seed")
    metrics = row.get("metrics")
    per_task = row.get("per_task")
    task_groups = row.get("task_groups", {})
    task_strata = row.get("task_strata", {})
    test_artifact_digest = row.get("test_artifact_digest")
    if not isinstance(method, str) or not method:
        raise ValueError("Main-results row method must be a non-empty string.")
    if not isinstance(trainable, bool):
        raise ValueError("Main-results row trainable must be a boolean.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Main-results row seed must be an integer.")
    if not isinstance(metrics, Mapping) or not isinstance(per_task, Mapping):
        raise ValueError("Main-results rows require metrics and per_task mappings.")
    if not isinstance(task_groups, Mapping) or not isinstance(task_strata, Mapping):
        raise ValueError("task_groups and task_strata must be mappings when provided.")
    if test_artifact_digest is not None and not isinstance(test_artifact_digest, str):
        raise ValueError("test_artifact_digest must be a string or null.")

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
    task_ids = set(normalized_tasks)
    normalized_groups = _normalize_task_values(task_groups, name="task_groups")
    if set(normalized_groups) - task_ids:
        raise ValueError("task_groups contains unknown task IDs.")
    normalized_groups = {
        task_id: normalized_groups.get(task_id, task_id) for task_id in task_ids
    }
    normalized_strata = _normalize_task_values(task_strata, name="task_strata")
    if set(normalized_strata) - task_ids:
        raise ValueError("task_strata contains unknown task IDs.")
    return _NormalizedRow(
        method=method,
        trainable=trainable,
        seed=seed,
        metrics=normalized_metrics,
        per_task=normalized_tasks,
        task_groups=normalized_groups,
        task_strata=normalized_strata,
        test_artifact_digest=test_artifact_digest,
    )


def _normalize_task_values(
    value: Mapping[object, object], *, name: str
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for task_id, item in value.items():
        if not isinstance(task_id, str) or not isinstance(item, str) or not item:
            raise ValueError(f"{name} must map task IDs to non-empty strings.")
        normalized[task_id] = item
    return normalized


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric.")
    return float(value)


__all__ = ["analyze_main_results"]
