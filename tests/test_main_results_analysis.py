from __future__ import annotations

from typing import Any, cast

import pytest

from graph_memory.analysis import analyze_main_results


def _row(
    method: str,
    seed: int,
    values: list[float],
    *,
    trainable: bool,
    groups: list[str] | None = None,
    strata: list[str] | None = None,
    digest: str | None = None,
) -> dict[str, object]:
    task_ids = [f"task-{index}" for index in range(len(values))]
    return {
        "method": method,
        "trainable": trainable,
        "seed": seed,
        "metrics": {"Full Support@2048 Tokens": sum(values) / len(values)},
        "per_task": {
            task_id: {"Full Support@2048 Tokens": value}
            for task_id, value in zip(task_ids, values, strict=True)
        },
        "task_groups": (
            dict(zip(task_ids, groups, strict=True)) if groups is not None else {}
        ),
        "task_strata": (
            dict(zip(task_ids, strata, strict=True)) if strata is not None else {}
        ),
        "test_artifact_digest": digest,
    }


def test_trainable_mean_std_and_deterministic_single_value() -> None:
    rows = [
        _row("bm25", 13, [1.0, 0.0], trainable=False),
        _row("dense_ft", 13, [1.0, 1.0], trainable=True),
        _row("dense_ft", 17, [0.0, 1.0], trainable=True),
    ]

    result = cast(dict[str, Any], analyze_main_results(rows, baseline_method="bm25"))
    summary = result["summary"]

    assert summary["bm25"]["metrics"]["Full Support@2048 Tokens"] == {"value": 0.5}
    dense = summary["dense_ft"]["metrics"]["Full Support@2048 Tokens"]
    assert dense["mean"] == 0.75
    assert dense["std"] > 0.0
    assert summary["dense_ft"]["seeds"] == [13, 17]
    assert result["test_task_count"] == 2
    assert result["test_cluster_count"] == 2
    assert result["cluster_unit"] == "task"


def test_paired_delta_is_method_minus_baseline_and_averages_seeds() -> None:
    rows = [
        _row("bm25", 13, [1.0, 0.0], trainable=False),
        _row("dense_ft", 13, [0.0, 0.0], trainable=True),
        _row("dense_ft", 17, [0.0, 1.0], trainable=True),
    ]

    result = cast(
        dict[str, Any],
        analyze_main_results(rows, baseline_method="bm25", bootstrap_samples=200),
    )
    paired = result["paired_analysis"]["dense_ft"]["metrics"][
        "Full Support@2048 Tokens"
    ]
    # Seed-averaged method-minus-baseline deltas are task0=-1 and task1=+0.5.
    assert paired["paired_query_count"] == 2
    assert paired["seed_pair_count"] == 2
    assert paired["mean_delta"] == pytest.approx(-0.25)
    assert paired["delta_direction"] == "method_minus_baseline"
    assert len(paired["ci_95"]) == 2

    inverted = cast(
        dict[str, Any],
        analyze_main_results(
            rows,
            baseline_method="bm25",
            bootstrap_samples=200,
            delta_direction="baseline_minus_method",
        ),
    )
    inverted_paired = inverted["paired_analysis"]["dense_ft"]["metrics"][
        "Full Support@2048 Tokens"
    ]
    assert inverted["delta_direction"] == "baseline_minus_method"
    assert inverted_paired["mean_delta"] == pytest.approx(0.25)
    assert inverted_paired["delta_direction"] == "baseline_minus_method"


def test_cluster_bootstrap_resamples_trajectory_groups_and_reports_strata() -> None:
    groups = ["trajectory-a", "trajectory-a", "trajectory-b"]
    strata = ["linked_recall", "linked_recall", "direct_recall"]
    rows = [
        _row(
            "dense_ft",
            13,
            [0.0, 0.0, 1.0],
            trainable=True,
            groups=groups,
            strata=strata,
            digest="test-digest",
        ),
        _row(
            "rgcn",
            13,
            [1.0, 1.0, 0.0],
            trainable=True,
            groups=groups,
            strata=strata,
            digest="test-digest",
        ),
    ]

    result = cast(
        dict[str, Any],
        analyze_main_results(rows, baseline_method="dense_ft", bootstrap_samples=200),
    )
    paired = result["paired_analysis"]["rgcn"]["metrics"]["Full Support@2048 Tokens"]
    assert result["test_cluster_count"] == 2
    assert result["cluster_unit"] == "group"
    assert result["test_artifact_digest"] == "test-digest"
    assert paired["paired_cluster_count"] == 2
    assert paired["paired_query_count"] == 3
    assert result["stratified_summary"]["rgcn"]["linked_recall"]["task_count"] == 2
    linked = result["stratified_paired_analysis"]["rgcn"]["linked_recall"]["metrics"][
        "Full Support@2048 Tokens"
    ]
    assert linked["mean_delta"] == 1.0
    assert linked["paired_cluster_count"] == 1
