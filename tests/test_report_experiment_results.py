from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from scripts.deliver.report_experiment_results import main, render_report


def _analysis() -> dict[str, Any]:
    return {
        "baseline_method": "full_rgcn",
        "delta_direction": "baseline_minus_method",
        "test_task_count": 2000,
        "test_cluster_count": 1207,
        "cluster_unit": "group",
        "test_artifact_digest": "fixed-test",
        "summary": {
            "full_rgcn": {
                "trainable": True,
                "seeds": [13, 17, 29],
                "metrics": {
                    "Recall@5": {"mean": 0.7869, "std": 0.006},
                    "Full Support@2048 Tokens": {"mean": 0.7802, "std": 0.0024},
                },
            },
            "random_edges": {
                "trainable": True,
                "seeds": [13, 17, 29],
                "metrics": {
                    "Recall@5": {"mean": 0.7165, "std": 0.0055},
                    "Full Support@2048 Tokens": {"mean": 0.7268, "std": 0.008},
                },
            },
        },
        "paired_analysis": {
            "random_edges": {
                "metrics": {
                    "Recall@5": {
                        "mean_delta": 0.0704,
                        "ci_95": [0.0542, 0.0872],
                    },
                    "Full Support@2048 Tokens": {
                        "mean_delta": 0.0533,
                        "ci_95": [0.0369, 0.0699],
                    },
                }
            }
        },
        "stratified_paired_analysis": {
            "random_edges": {
                "direct_recall": {
                    "metrics": {
                        "Recall@5": {
                            "mean_delta": 0.01,
                            "ci_95": [-0.01, 0.03],
                        },
                        "Full Support@2048 Tokens": {
                            "mean_delta": 0.012,
                            "ci_95": [-0.008, 0.034],
                        },
                    }
                },
                "multi_fact_recall": {
                    "metrics": {
                        "Recall@5": {
                            "mean_delta": 0.12,
                            "ci_95": [0.08, 0.16],
                        },
                        "Full Support@2048 Tokens": {
                            "mean_delta": 0.121,
                            "ci_95": [0.082, 0.16],
                        },
                    }
                },
            }
        },
    }


def _write_run(root: Path, *, variant: str, seed: int, epoch_zero: bool) -> None:
    (root / "workflow").mkdir(parents=True)
    (root / "training").mkdir(parents=True)
    (root / "workflow" / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "method": "provenance_rgcn",
                "variant": variant,
                "dataset": "isetrace",
                "profile": "full",
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "epoch": 1,
            "initial_selection_metric_value": 0.72,
            "selection_metric_value": 0.68 if epoch_zero else 0.75,
            "best_dev_metric": 0.72 if epoch_zero else 0.75,
        }
    ]
    (root / "training" / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "training" / "control_diagnostics.json").write_text(
        json.dumps(
            {
                "train": {
                    "physical_edge_count": 100,
                    "active_edge_count": 100,
                    "rewired_edge_count": 90 if variant == "random_edges" else 0,
                    "changed_graph_count": 10 if variant == "random_edges" else 0,
                }
            }
        ),
        encoding="utf-8",
    )


def test_render_report_formats_percentages_intervals_and_strata() -> None:
    report = render_report(
        _analysis(),
        title="E2 relation controls",
        metrics=("Recall@5", "Full Support@2048 Tokens"),
    )

    assert "# E2 relation controls" in report
    assert "78.69 ± 0.60" in report
    assert "+7.04 [+5.42, +8.72] ↑" in report
    assert "+1.00 [-1.00, +3.00] ≈" in report
    assert "multi_fact_recall" in report


def test_cli_discovers_canonical_prefix_and_adds_diagnostics(tmp_path: Path) -> None:
    input_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(_analysis()), encoding="utf-8")
    run_root = tmp_path / "runs"
    _write_run(
        run_root / "isetrace_v7_e2rel_rgcn_full_s13",
        variant="full_rgcn",
        seed=13,
        epoch_zero=False,
    )
    _write_run(
        run_root / "isetrace_v7_e2rel_rgcn_randedge_s13",
        variant="random_edges",
        seed=13,
        epoch_zero=True,
    )
    output = tmp_path / "report.md"

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--run-root",
                str(run_root),
                "--name-prefix",
                "isetrace_v7_e2rel_",
                "--metrics",
                "Recall@5,Full Support@2048 Tokens",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = output.read_text(encoding="utf-8")
    assert "## Training and control diagnostics" in report
    assert "| random_edges | 13 | 0 | 0.7200 | 100 | 100 | 90 | 10 |" in report
    assert "`random_edges` selected the initial epoch-0 checkpoint" in report


def test_canonical_abbreviated_labels_match_run_diagnostics(tmp_path: Path) -> None:
    analysis = _analysis()
    analysis["summary"] = {
        "rgcn_full": analysis["summary"]["full_rgcn"],
        "rgcn_randedge": analysis["summary"]["random_edges"],
    }
    analysis["baseline_method"] = "rgcn_full"
    analysis["paired_analysis"] = {
        "rgcn_randedge": analysis["paired_analysis"]["random_edges"]
    }
    analysis["stratified_paired_analysis"] = {
        "rgcn_randedge": analysis["stratified_paired_analysis"]["random_edges"]
    }
    input_path = tmp_path / "analysis.json"
    input_path.write_text(json.dumps(analysis), encoding="utf-8")
    run_root = tmp_path / "runs"
    _write_run(
        run_root / "isetrace_v7_e2rel_rgcn_full_s13",
        variant="full_rgcn",
        seed=13,
        epoch_zero=False,
    )
    _write_run(
        run_root / "isetrace_v7_e2rel_rgcn_randedge_s13",
        variant="random_edges",
        seed=13,
        epoch_zero=True,
    )
    output = tmp_path / "report.md"

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--run-root",
                str(run_root),
                "--name-prefix",
                "isetrace_v7_e2rel_",
                "--metrics",
                "Recall@5",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = output.read_text(encoding="utf-8")
    assert "| rgcn_full | 13 | 1 | 0.7500 |" in report
    assert "| rgcn_randedge | 13 | 0 | 0.7200 |" in report
