from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scripts.aggregate_main_results import main


def _write_run(
    root: Path,
    *,
    method: str,
    seed: int,
    coverage: tuple[float, float],
) -> None:
    (root / "workflow").mkdir(parents=True)
    (root / "metrics").mkdir(parents=True)
    (root / "assets").mkdir(parents=True)
    (root / "workflow" / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "method": method,
                "variant": None,
                "dataset": "isetrace",
                "profile": "full",
                "seed": seed,
            },
            sort_keys=False,
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with (root / "metrics" / "per_task.jsonl").open("w", encoding="utf-8") as stream:
        for index, value in enumerate(coverage):
            stream.write(
                json.dumps(
                    {
                        "task_id": f"q{index}",
                        "Coverage@256 Tokens": value / 4,
                        "Coverage@512 Tokens": value / 2,
                        "Coverage@1024 Tokens": value,
                        "Coverage@2048 Tokens": value,
                        "Coverage@4096 Tokens": value,
                        "Coverage@8192 Tokens": value,
                        "Full Support@256 Tokens": value / 4,
                        "Full Support@512 Tokens": value / 2,
                        "Full Support@1024 Tokens": value,
                        "Full Support@2048 Tokens": value,
                        "Full Support@4096 Tokens": value,
                        "Full Support@8192 Tokens": value,
                        "Coverage Budget-AUC": value,
                        "Full Support Budget-AUC": value,
                        "Recall@5": value,
                        "MRR": value,
                    }
                )
                + "\n"
            )


def test_cli_aggregates_token_metrics_and_legacy_trajectory_metadata(
    tmp_path: Path,
) -> None:
    dense = tmp_path / "dense"
    rgcn = tmp_path / "rgcn"
    _write_run(dense, method="dense_ft", seed=13, coverage=(0.0, 1.0))
    _write_run(rgcn, method="provenance_rgcn", seed=13, coverage=(1.0, 1.0))
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "query_id": "q0",
                        "trajectory_id": "trajectory-a",
                        "memory_mode": "linked_recall",
                    }
                ),
                json.dumps(
                    {
                        "query_id": "q1",
                        "trajectory_id": "trajectory-a",
                        "memory_mode": "linked_recall",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "main.json"
    output_csv = tmp_path / "main.csv"

    assert (
        main(
            [
                "--run",
                f"dense_natural={dense}",
                "--run",
                f"rgcn_natural={rgcn}",
                "--baseline",
                "dense_natural",
                "--trainable",
                "dense_natural",
                "--trainable",
                "rgcn_natural",
                "--query-metadata",
                str(metadata),
                "--expected-task-count",
                "2",
                "--delta-direction",
                "baseline-minus-method",
                "--bootstrap-samples",
                "100",
                "--output",
                str(output),
                "--output-csv",
                str(output_csv),
            ]
        )
        == 0
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["test_artifact_digest"] == "fixed-test-digest"
    assert result["test_task_count"] == 2
    assert result["test_cluster_count"] == 1
    assert (
        result["summary"]["rgcn_natural"]["metrics"]["Coverage@1024 Tokens"]["mean"]
        == 1.0
    )
    assert (
        result["summary"]["rgcn_natural"]["metrics"]["Full Support@8192 Tokens"]["mean"]
        == 1.0
    )
    paired = result["paired_analysis"]["rgcn_natural"]["metrics"][
        "Coverage@1024 Tokens"
    ]
    assert result["delta_direction"] == "baseline_minus_method"
    assert paired["mean_delta"] == -0.5
    assert paired["delta_direction"] == "baseline_minus_method"
    assert paired["paired_cluster_count"] == 1
    assert (
        result["stratified_summary"]["rgcn_natural"]["linked_recall"]["task_count"] == 2
    )
    with output_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["Method"] for row in rows} == {"dense_natural", "rgcn_natural"}
    rgcn_row = next(row for row in rows if row["Method"] == "rgcn_natural")
    assert float(rgcn_row["Coverage@1024 Tokens"]) == 1.0


def test_cli_filters_to_explicit_task_subset(tmp_path: Path) -> None:
    dense = tmp_path / "dense"
    rgcn = tmp_path / "rgcn"
    _write_run(dense, method="dense_ft", seed=13, coverage=(0.0, 1.0))
    _write_run(rgcn, method="provenance_rgcn", seed=13, coverage=(1.0, 1.0))
    task_ids = tmp_path / "task_ids.json"
    task_ids.write_text(json.dumps(["q0"]) + "\n", encoding="utf-8")
    output = tmp_path / "subset.json"

    assert (
        main(
            [
                "--run",
                f"dense_natural={dense}",
                "--run",
                f"rgcn_natural={rgcn}",
                "--baseline",
                "dense_natural",
                "--trainable",
                "dense_natural",
                "--trainable",
                "rgcn_natural",
                "--task-ids",
                str(task_ids),
                "--expected-task-count",
                "1",
                "--bootstrap-samples",
                "100",
                "--metric",
                "Coverage@1024 Tokens",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["test_task_count"] == 1
    assert result["task_subset"]["task_count"] == 1
    assert set(result["summary"]["dense_natural"]["metrics"]) == {
        "Coverage@1024 Tokens"
    }
    assert (
        result["summary"]["dense_natural"]["metrics"]["Coverage@1024 Tokens"]["mean"]
        == 0.0
    )
    assert (
        result["summary"]["rgcn_natural"]["metrics"]["Coverage@1024 Tokens"]["mean"]
        == 1.0
    )
    assert (
        result["paired_analysis"]["rgcn_natural"]["metrics"]["Coverage@1024 Tokens"][
            "mean_delta"
        ]
        == 1.0
    )
