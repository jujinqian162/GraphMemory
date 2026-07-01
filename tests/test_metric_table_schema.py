from __future__ import annotations

from pathlib import Path

import scripts.aggregate_tables as aggregate_tables
from graph_memory.evaluation.tables import metric_table_schema_for_suite, split_metric_tables


def test_split_metric_tables_uses_explicit_longmemeval_schema_for_empty_rows() -> None:
    schema = metric_table_schema_for_suite("longmemeval")

    main_rows, path_rows, efficiency_rows = split_metric_tables([], schema=schema)

    assert main_rows == []
    assert path_rows == []
    assert efficiency_rows == []
    assert "Turn Recall@5" in schema.main_columns
    assert "Evidence F1@10" not in schema.wide_columns


def test_aggregate_tables_writes_configured_longmemeval_columns_for_empty_inputs(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    output_main = tmp_path / "main_results.csv"
    output_path = tmp_path / "path_results.csv"
    output_efficiency = tmp_path / "efficiency_results.csv"

    assert aggregate_tables.main(
        [
            "--input_dir",
            str(metrics_dir),
            "--output_main",
            str(output_main),
            "--output_path",
            str(output_path),
            "--output_efficiency",
            str(output_efficiency),
            "--metric_suite",
            "longmemeval",
        ]
    ) == 0

    assert output_main.read_text(encoding="utf-8").splitlines()[0].startswith("Method,Turn Recall@5")
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "Method,Path Recall@10,Edge Recall@10"
    assert "Memory Size" in output_efficiency.read_text(encoding="utf-8").splitlines()[0]
