from __future__ import annotations

from typing import Any, cast

from graph_memory.analysis import analyze_provenance_ablation_rows


def test_paired_ablation_analysis_reports_uncertainty_and_discordance() -> None:
    rows = [
        _row("full_rgcn", 13, [1.0, 0.0]),
        _row("wo_edge_weight", 13, [0.0, 1.0]),
        _row("full_rgcn", 17, [1.0, 1.0]),
        _row("wo_edge_weight", 17, [1.0, 0.0]),
    ]

    result = cast(dict[str, Any], analyze_provenance_ablation_rows(rows, bootstrap_samples=100))
    paired = result["paired_analysis"]["wo_edge_weight"]

    assert len(result["seed_rows"]) == 4
    assert result["summary"]["full_rgcn"]["Full Support@5"]["mean"] == 0.75
    assert paired["metrics"]["Full Support@5"]["paired_query_count"] == 4
    assert len(paired["metrics"]["Full Support@5"]["paired_ci_95"]) == 2
    assert paired["discordant_full_support"]["Full Support@5"] == {
        "full_only": 2,
        "ablation_only": 1,
    }


def _row(variant: str, seed: int, values: list[float]) -> dict[str, object]:
    return {
        "variant": variant,
        "seed": seed,
        "metrics": {"Full Support@5": sum(values) / len(values)},
        "per_task": {
            f"task-{index}": {"Full Support@5": value}
            for index, value in enumerate(values)
        },
        "identities": {"dataset": "v3"},
    }
