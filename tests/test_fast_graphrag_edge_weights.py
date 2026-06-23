from __future__ import annotations

import math

from graph_memory.retrieval.methods.fast_graphrag.edge_weights import pmi_edge_weight


def test_pmi_edge_weight_matches_official_formula() -> None:
    weight = pmi_edge_weight(
        edge_count=2,
        total_edge_weights=10,
        source_frequency=4,
        target_frequency=5,
        total_frequency_occurrences=20,
    )

    assert math.isclose(weight, 0.4, rel_tol=1e-12)


def test_pmi_edge_weight_returns_zero_for_empty_denominator() -> None:
    assert pmi_edge_weight(
        edge_count=0,
        total_edge_weights=0,
        source_frequency=0,
        target_frequency=0,
        total_frequency_occurrences=0,
    ) == 0.0
