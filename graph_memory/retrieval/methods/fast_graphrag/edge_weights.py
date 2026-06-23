from __future__ import annotations

import math


def pmi_edge_weight(
    *,
    edge_count: int,
    total_edge_weights: int,
    source_frequency: int,
    target_frequency: int,
    total_frequency_occurrences: int,
) -> float:
    if (
        edge_count <= 0
        or total_edge_weights <= 0
        or source_frequency <= 0
        or target_frequency <= 0
        or total_frequency_occurrences <= 0
    ):
        return 0.0
    prop_weight = edge_count / total_edge_weights
    source_prop = source_frequency / total_frequency_occurrences
    target_prop = target_frequency / total_frequency_occurrences
    denominator = source_prop * target_prop
    if denominator <= 0.0:
        return 0.0
    return prop_weight * math.log2(prop_weight / denominator)


__all__ = ["pmi_edge_weight"]
