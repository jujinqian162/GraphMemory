from __future__ import annotations

import random
import statistics
from collections.abc import Sequence


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
    """95% paired bootstrap interval over per-query paired ``deltas``.

    ``deltas`` are the per-query baseline-minus-method differences already
    aligned by ``task_id``. Returns the mean delta, the interval bounds, and the
    number of paired queries.
    """
    mean_delta, lower, upper = bootstrap_ci(deltas, samples=samples, seed=seed)
    return {
        "mean_delta": mean_delta,
        "ci_95": [lower, upper],
        "paired_query_count": len(deltas),
    }


__all__ = ["bootstrap_ci", "paired_delta_bootstrap_ci"]
