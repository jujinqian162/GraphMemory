from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from graph_memory.analysis.paired_bootstrap import bootstrap_ci


@dataclass(frozen=True)
class _NormalizedRow:
    variant: str
    seed: int
    metrics: dict[str, float]
    per_task: dict[str, dict[str, float]]
    identities: dict[str, object]


def analyze_provenance_ablation_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_variant: str = "full_rgcn",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 13,
) -> dict[str, object]:
    normalized = [_normalize_row(row) for row in rows]
    by_variant_seed = {
        (row.variant, row.seed): row for row in normalized
    }
    if len(by_variant_seed) != len(normalized):
        raise ValueError("Ablation analysis requires unique (variant, seed) rows.")
    variants = sorted({row.variant for row in normalized})
    seeds = sorted({row.seed for row in normalized})
    if baseline_variant not in variants:
        raise ValueError(f"Missing baseline variant={baseline_variant!r}.")
    metric_names = sorted(
        set.intersection(*(set(row.metrics) for row in normalized))
    )
    seed_rows = [
        {
            "variant": row.variant,
            "seed": row.seed,
            **row.metrics,
            "identities": row.identities,
        }
        for row in sorted(normalized, key=lambda item: (item.variant, item.seed))
    ]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for variant in variants:
        variant_rows = [row for row in normalized if row.variant == variant]
        summary[variant] = {
            metric: _mean_std([row.metrics[metric] for row in variant_rows])
            for metric in metric_names
        }

    paired: dict[str, object] = {}
    for variant in variants:
        if variant == baseline_variant:
            continue
        paired_seeds = [
            seed
            for seed in seeds
            if (baseline_variant, seed) in by_variant_seed
            and (variant, seed) in by_variant_seed
        ]
        if not paired_seeds:
            raise ValueError(f"No paired seeds for variant={variant!r}.")
        metric_results: dict[str, object] = {}
        discordant: dict[str, dict[str, int]] = {}
        for metric in metric_names:
            query_deltas: list[float] = []
            baseline_only = 0
            ablation_only = 0
            for seed in paired_seeds:
                baseline = by_variant_seed[(baseline_variant, seed)]
                ablation = by_variant_seed[(variant, seed)]
                baseline_tasks = baseline.per_task
                ablation_tasks = ablation.per_task
                if set(baseline_tasks) != set(ablation_tasks):
                    raise ValueError(
                        f"Paired task IDs differ for variant={variant!r}, seed={seed}."
                    )
                for task_id in sorted(baseline_tasks):
                    baseline_value = baseline_tasks[task_id].get(metric)
                    ablation_value = ablation_tasks[task_id].get(metric)
                    if baseline_value is None or ablation_value is None:
                        continue
                    query_deltas.append(baseline_value - ablation_value)
                    if metric.startswith("Full Support@"):
                        baseline_only += int(baseline_value == 1.0 and ablation_value == 0.0)
                        ablation_only += int(baseline_value == 0.0 and ablation_value == 1.0)
            mean_delta, lower, upper = bootstrap_ci(
                query_deltas,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            metric_results[metric] = {
                "mean_full_minus_ablation": mean_delta,
                "paired_ci_95": [lower, upper],
                "paired_query_count": len(query_deltas),
            }
            if metric.startswith("Full Support@"):
                discordant[metric] = {
                    "full_only": baseline_only,
                    "ablation_only": ablation_only,
                }
        paired[variant] = {
            "seeds": paired_seeds,
            "metrics": metric_results,
            "discordant_full_support": discordant,
        }
    return {
        "baseline_variant": baseline_variant,
        "seed_rows": seed_rows,
        "summary": summary,
        "paired_analysis": paired,
    }


def _normalize_row(row: Mapping[str, object]) -> _NormalizedRow:
    variant = row.get("variant")
    seed = row.get("seed")
    metrics = row.get("metrics")
    per_task = row.get("per_task")
    identities = row.get("identities", {})
    if not isinstance(variant, str) or not variant:
        raise ValueError("Ablation row variant must be a non-empty string.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Ablation row seed must be an integer.")
    if not isinstance(metrics, Mapping) or not isinstance(per_task, Mapping):
        raise ValueError("Ablation rows require metrics and per_task mappings.")
    normalized_metrics = {
        str(name): _number(value, f"metrics.{name}")
        for name, value in metrics.items()
    }
    normalized_tasks: dict[str, dict[str, float]] = {}
    for task_id, raw_task_metrics in per_task.items():
        if not isinstance(task_id, str) or not isinstance(raw_task_metrics, Mapping):
            raise ValueError("per_task must map task IDs to metric mappings.")
        normalized_tasks[task_id] = {
            str(name): _number(value, f"per_task.{task_id}.{name}")
            for name, value in raw_task_metrics.items()
        }
    normalized_identities = (
        {str(name): value for name, value in identities.items()}
        if isinstance(identities, Mapping)
        else {}
    )
    return _NormalizedRow(
        variant=variant,
        seed=seed,
        metrics=normalized_metrics,
        per_task=normalized_tasks,
        identities=normalized_identities,
    )


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be numeric.")
    return float(value)


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


__all__ = ["analyze_provenance_ablation_rows"]
