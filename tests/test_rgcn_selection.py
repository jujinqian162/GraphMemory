from __future__ import annotations

import pytest

from graph_memory.models.graph_retriever.selection import (
    RgcnSelectionMetric,
    RgcnSelectionSettings,
    is_selection_improvement,
    resolve_selection_metric,
)


def test_selection_resolves_composite_and_named_metrics() -> None:
    metrics: dict[RgcnSelectionMetric, float] = {
        "dev_composite": 0.61,
        "dev_full_support_at_5": 0.5,
        "dev_full_support_at_10": 0.7,
        "dev_recall_at_5": 0.8,
        "dev_mrr": 0.9,
        "dev_loss": 1.2,
    }

    assert resolve_selection_metric(metrics, RgcnSelectionSettings()) == pytest.approx(
        0.61
    )
    assert resolve_selection_metric(
        metrics,
        RgcnSelectionSettings(best_metric="dev_full_support_at_10"),
    ) == pytest.approx(0.7)
    assert resolve_selection_metric(
        metrics,
        RgcnSelectionSettings(best_metric="dev_loss", higher_is_better=False),
    ) == pytest.approx(1.2)


def test_selection_comparison_honors_direction() -> None:
    assert is_selection_improvement(
        current=0.8,
        best=0.7,
        settings=RgcnSelectionSettings(higher_is_better=True),
    )
    assert not is_selection_improvement(
        current=0.6,
        best=0.7,
        settings=RgcnSelectionSettings(higher_is_better=True),
    )
    assert is_selection_improvement(
        current=0.6,
        best=0.7,
        settings=RgcnSelectionSettings(best_metric="dev_loss", higher_is_better=False),
    )
    assert not is_selection_improvement(
        current=0.8,
        best=0.7,
        settings=RgcnSelectionSettings(best_metric="dev_loss", higher_is_better=False),
    )


def test_selection_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="dev_mrr"):
        resolve_selection_metric(
            {"dev_loss": 1.0},
            RgcnSelectionSettings(best_metric="dev_mrr"),
        )
