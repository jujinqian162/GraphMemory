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
) -> dict[str, object]:
    return {
        "method": method,
        "trainable": trainable,
        "seed": seed,
        "metrics": {"Full Support@5": sum(values) / len(values)},
        "per_task": {
            f"task-{index}": {"Full Support@5": value}
            for index, value in enumerate(values)
        },
    }


def test_trainable_mean_std_and_deterministic_single_value() -> None:
    rows = [
        _row("bm25", 13, [1.0, 0.0], trainable=False),
        _row("dense_ft", 13, [1.0, 1.0], trainable=True),
        _row("dense_ft", 17, [0.0, 1.0], trainable=True),
    ]

    result = cast(dict[str, Any], analyze_main_results(rows, baseline_method="bm25"))
    summary = result["summary"]

    assert summary["bm25"]["metrics"]["Full Support@5"] == {"value": 0.5}
    dense = summary["dense_ft"]["metrics"]["Full Support@5"]
    assert dense["mean"] == 0.75
    assert dense["std"] > 0.0
    assert summary["dense_ft"]["seeds"] == [13, 17]
    assert result["test_task_count"] == 2


def test_paired_ci_against_baseline() -> None:
    rows = [
        _row("bm25", 13, [1.0, 0.0], trainable=False),
        _row("dense_ft", 13, [0.0, 0.0], trainable=True),
        _row("dense_ft", 17, [0.0, 1.0], trainable=True),
    ]

    result = cast(
        dict[str, Any],
        analyze_main_results(rows, baseline_method="bm25", bootstrap_samples=200),
    )
    paired = result["paired_analysis"]["dense_ft"]["metrics"]["Full Support@5"]
    # baseline [1,0] vs dense seed13 [0,0] -> deltas 1,0; seed17 [0,1] -> 1,-1
    assert paired["paired_query_count"] == 4
    assert paired["mean_delta"] == pytest.approx(0.25)
    assert len(paired["ci_95"]) == 2


def test_mismatched_task_ids_raise() -> None:
    good = _row("bm25", 13, [1.0, 0.0], trainable=False)
    bad = {
        "method": "dense_ft",
        "trainable": True,
        "seed": 13,
        "metrics": {"Full Support@5": 1.0},
        "per_task": {"task-0": {"Full Support@5": 1.0}},  # only one task
    }
    with pytest.raises(ValueError, match="share the same test split"):
        analyze_main_results([good, bad], baseline_method="bm25")


def test_inconsistent_trainable_flag_raises() -> None:
    rows = [
        _row("dense_ft", 13, [1.0, 0.0], trainable=True),
        _row("dense_ft", 17, [1.0, 0.0], trainable=False),
    ]
    with pytest.raises(ValueError, match="inconsistent trainable"):
        analyze_main_results(rows, baseline_method="dense_ft")
