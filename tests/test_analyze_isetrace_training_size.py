from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import pytest
import yaml

import scripts.analyze_isetrace_training_size as training_size
from scripts.aggregate_main_results import ISETRACE_MAIN_METRICS


def _write_run(
    root: Path,
    *,
    method: str,
    variant: str,
    seed: int,
    train_trajectories: int,
    dev_trajectories: int,
    test_trajectories: int,
    coverage: tuple[float, float],
    support: tuple[float, float],
) -> None:
    (root / "workflow").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "assets").mkdir()
    (root / "metrics").mkdir()
    (root / "workflow" / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "method": method,
                "variant": variant,
                "dataset": "isetrace",
                "profile": "full",
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )
    (root / "config" / "resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "isetrace",
                    "trajectories": {
                        "splits": {
                            "train": train_trajectories,
                            "dev": dev_trajectories,
                            "test": test_trajectories,
                        }
                    },
                },
                "split_seed": 13,
            }
        ),
        encoding="utf-8",
    )
    (root / "assets" / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "assets": [
                    {
                        "kind": "dataset",
                        "digest": "fixed-test-digest",
                        "origin": {"split": "test"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with (root / "metrics" / "per_task.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for index, (coverage_value, support_value) in enumerate(
            zip(coverage, support, strict=True)
        ):
            row = {
                "task_id": f"q{index}",
                "graph_id": f"g{index}",
                **{
                    metric: (
                        support_value
                        if metric.startswith("Full Support")
                        else coverage_value
                    )
                    for metric in ISETRACE_MAIN_METRICS
                },
            }
            stream.write(json.dumps(row) + "\n")


def test_training_size_analysis_aggregates_cohort_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_size, "TRAINING_SIZES", ((25, 1), (100, 4)))
    monkeypatch.setattr(training_size, "MODEL_SEEDS", (13, 17))
    monkeypatch.setattr(training_size, "DEV_TRAJECTORIES", 1)
    monkeypatch.setattr(training_size, "TEST_TRAJECTORIES", 2)
    monkeypatch.setattr(training_size, "TEST_TASKS", 2)
    run_root = tmp_path / "runs"
    for percentage, train_count in training_size.TRAINING_SIZES:
        for _, (token, method, variant) in training_size.METHODS.items():
            for seed in training_size.MODEL_SEEDS:
                seed_offset = 0.01 if seed == 17 else 0.0
                base = 0.4 + percentage / 1000.0 + seed_offset
                residual = 0.1 if variant == "full_rgcn" else 0.0
                _write_run(
                    run_root
                    / f"isetrace_v7_trainsize{percentage}_{token}_s{seed}",
                    method=method,
                    variant=variant,
                    seed=seed,
                    train_trajectories=train_count,
                    dev_trajectories=1,
                    test_trajectories=2,
                    coverage=(base + residual, base + residual),
                    support=(base + residual / 2, base + residual / 2),
                )
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": f"q{index}",
                    "trajectory_id": f"g{index}",
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "analysis.json"
    output_csv = tmp_path / "analysis.csv"
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.pdf"

    assert (
        training_size.main(
            [
                "--run-root",
                str(run_root),
                "--query-metadata",
                str(metadata),
                "--output",
                str(output),
                "--output-csv",
                str(output_csv),
                "--report",
                str(report),
                "--figure",
                str(figure),
                "--bootstrap-samples",
                "100",
            ]
        )
        == 0
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    points = cast(list[dict[str, object]], result["points"])
    assert [point["train_trajectory_count"] for point in points] == [1, 4]
    first_effects = cast(dict[str, dict[str, object]], points[0]["effects_vs_exact_seed"])
    residual = cast(
        dict[str, dict[str, object]], first_effects["residual_rgcn"]["metrics"]
    )
    dense = cast(
        dict[str, dict[str, object]],
        first_effects["dense_ft_provenance_unit"]["metrics"],
    )
    assert residual["Coverage@1024 Tokens"]["mean_delta"] == pytest.approx(0.1)
    assert residual["Full Support@2048 Tokens"]["mean_delta"] == pytest.approx(0.05)
    assert dense["Coverage@1024 Tokens"]["mean_delta"] == pytest.approx(0.0)
    with output_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert "Residual gain" in report.read_text(encoding="utf-8")
    assert figure.stat().st_size > 0


def test_training_size_analysis_rejects_wrong_resolved_split(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(
        run,
        method="dense_ft",
        variant="provenance_unit",
        seed=13,
        train_trajectories=242,
        dev_trajectories=352,
        test_trajectories=1207,
        coverage=(0.4, 0.5),
        support=(0.3, 0.4),
    )

    with pytest.raises(ValueError, match="trajectory split mismatch"):
        training_size._validate_run(
            run,
            method="dense_ft",
            variant="provenance_unit",
            seed=13,
            train_trajectories=241,
        )
