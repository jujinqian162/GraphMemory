from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence


def bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile bootstrap for the mean of ``values``.

    Returns ``(mean, lower_2.5%, upper_97.5%)``. Empty input yields all zeros so
    callers can treat a missing population as a degenerate interval rather than
    raising.
    """
    if not values:
        return 0.0, 0.0, 0.0
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    return statistics.fmean(values), means[lower_index], means[upper_index]


def paired_delta_bootstrap_ci(
    deltas: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """95% paired bootstrap interval over already aligned task deltas."""
    mean_delta, lower, upper = bootstrap_ci(deltas, samples=samples, seed=seed)
    return {
        "mean_delta": mean_delta,
        "ci_95": [lower, upper],
        "paired_query_count": len(deltas),
    }


def paired_cluster_delta_bootstrap_ci(
    deltas_by_cluster: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Paired percentile bootstrap that resamples whole query clusters.

    Each cluster is sampled with replacement and contributes all of its task
    deltas. This preserves within-cluster dependence, such as multiple ISETrace
    queries authored from the same trajectory.
    """
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    normalized = {
        cluster_id: tuple(float(value) for value in values)
        for cluster_id, values in deltas_by_cluster.items()
        if values
    }
    if not normalized:
        return {
            "mean_delta": 0.0,
            "ci_95": [0.0, 0.0],
            "paired_query_count": 0,
            "paired_cluster_count": 0,
        }
    cluster_ids = sorted(normalized)
    observed = [value for cluster_id in cluster_ids for value in normalized[cluster_id]]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [value for cluster_id in sampled for value in normalized[cluster_id]]
        means.append(statistics.fmean(values))
    means.sort()
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    return {
        "mean_delta": statistics.fmean(observed),
        "ci_95": [means[lower_index], means[upper_index]],
        "paired_query_count": len(observed),
        "paired_cluster_count": len(cluster_ids),
    }


__all__ = [
    "bootstrap_ci",
    "paired_cluster_delta_bootstrap_ci",
    "paired_delta_bootstrap_ci",
]
