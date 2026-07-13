from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias


RgcnSelectionMetric: TypeAlias = Literal[
    "dev_composite",
    "dev_full_support_at_5",
    "dev_full_support_at_10",
    "dev_recall_at_5",
    "dev_mrr",
    "dev_loss",
]
SUPPORTED_RGCN_SELECTION_METRICS: tuple[RgcnSelectionMetric, ...] = (
    "dev_composite",
    "dev_full_support_at_5",
    "dev_full_support_at_10",
    "dev_recall_at_5",
    "dev_mrr",
    "dev_loss",
)


@dataclass(frozen=True)
class RgcnSelectionSettings:
    best_metric: RgcnSelectionMetric = "dev_composite"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.best_metric not in SUPPORTED_RGCN_SELECTION_METRICS:
            raise ValueError(f"Unsupported R-GCN selection metric: {self.best_metric}")


def build_selection_metrics(
    *,
    dev_full_support_at_5: float,
    dev_full_support_at_10: float,
    dev_recall_at_5: float,
    dev_mrr: float,
    dev_loss: float,
) -> dict[RgcnSelectionMetric, float]:
    return {
        "dev_composite": (
            0.50 * dev_full_support_at_5 + 0.30 * dev_recall_at_5 + 0.20 * dev_mrr
        ),
        "dev_full_support_at_5": dev_full_support_at_5,
        "dev_full_support_at_10": dev_full_support_at_10,
        "dev_recall_at_5": dev_recall_at_5,
        "dev_mrr": dev_mrr,
        "dev_loss": dev_loss,
    }


def resolve_selection_metric(
    metrics: Mapping[RgcnSelectionMetric, float], settings: RgcnSelectionSettings
) -> float:
    try:
        selected = float(metrics[settings.best_metric])
    except KeyError as error:
        raise ValueError(
            f"R-GCN selection metric is missing: {settings.best_metric}"
        ) from error
    if not math.isfinite(selected):
        raise ValueError(
            f"R-GCN selection metric must be finite: {settings.best_metric}={selected}"
        )
    return selected


def is_selection_improvement(
    *, current: float, best: float, settings: RgcnSelectionSettings
) -> bool:
    return current > best if settings.higher_is_better else current < best


__all__ = [
    "RgcnSelectionMetric",
    "RgcnSelectionSettings",
    "SUPPORTED_RGCN_SELECTION_METRICS",
    "build_selection_metrics",
    "is_selection_improvement",
    "resolve_selection_metric",
]
