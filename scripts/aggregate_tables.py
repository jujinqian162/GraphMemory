from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.evaluation.tables import (
    EFFICIENCY_RESULT_COLUMNS,
    MAIN_RESULT_COLUMNS,
    PATH_RESULT_COLUMNS,
    split_metric_tables,
)
from graph_memory.contracts.common import JsonValue
from graph_memory.contracts.metrics import MetricRow
from graph_memory.io import read_csv, read_json, write_csv
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary

LOGGER = logging.getLogger("aggregate_tables")
ABLATION_RESULT_COLUMNS = [
    "Method",
    "Variant",
    "Recall@5",
    "Full Support@5",
    "Connected Evidence Recall@10",
    "Path Recall@10",
    "Retrieval Latency / Query",
]


@dataclass(frozen=True)
class AggregateTablesArgs:
    input_dir: str
    output_main: str
    output_path: str
    output_efficiency: str
    ablation_index: str | None = None
    output_ablation: str | None = None
    ablation_selections: tuple[str, ...] = ()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_main = Path(args.output_main)
    summary_path = output_main.with_name("aggregate_tables.run_summary.json")
    output_paths = {Path(args.output_main), Path(args.output_path), Path(args.output_efficiency)}
    inputs: dict[str, JsonValue] = {"input_dir": args.input_dir}
    outputs = {
        "main": args.output_main,
        "path": args.output_path,
        "efficiency": args.output_efficiency,
        "run_summary": str(summary_path),
    }
    if args.ablation_index is not None and args.output_ablation is not None:
        inputs["ablation_index"] = args.ablation_index
        inputs["ablation_selections"] = list(args.ablation_selections)
        outputs["ablation"] = args.output_ablation

    try:
        metric_files = [
            path
            for path in sorted(Path(args.input_dir).glob("*.csv"))
            if path not in output_paths and _looks_like_metric_file(path)
        ]
        rows: list[MetricRow] = []
        for metric_file in metric_files:
            rows.extend(cast(list[MetricRow], read_csv(metric_file)))
        main_rows, path_rows, efficiency_rows = split_metric_tables(rows)
        write_csv(args.output_main, main_rows, MAIN_RESULT_COLUMNS)
        write_csv(args.output_path, path_rows, PATH_RESULT_COLUMNS)
        write_csv(args.output_efficiency, efficiency_rows, EFFICIENCY_RESULT_COLUMNS)
        ablation_rows = (
            _indexed_ablation_rows(args.ablation_index, args.ablation_selections)
            if args.ablation_index is not None
            else []
        )
        if args.output_ablation is not None:
            write_csv(args.output_ablation, ablation_rows, ABLATION_RESULT_COLUMNS)

        summary = build_run_summary(
            script="aggregate_tables.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config={},
            inputs=inputs,
            outputs=outputs,
            counts={"metric_files": len(metric_files), "rows": len(rows)},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[f"included={metric_file}" for metric_file in metric_files],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote main results: %s", args.output_main)
        LOGGER.info("wrote path results: %s", args.output_path)
        LOGGER.info("wrote efficiency results: %s", args.output_efficiency)
        if args.output_ablation is not None:
            LOGGER.info("wrote ablation results: %s", args.output_ablation)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="aggregate_tables.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config={},
            inputs=inputs,
            outputs=outputs,
            counts={},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
            error=str(error),
        )
        write_run_summary(summary_path, summary)
        raise


def _looks_like_metric_file(path: Path) -> bool:
    return (
        path.name.startswith("main_results_")
        or path.name.startswith("metrics_")
        or path.name.endswith(".metrics.csv")
    )


def _indexed_ablation_rows(index_path: str, selection_values: Sequence[str] = ()) -> list[dict[str, object]]:
    index = read_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("metrics"), list):
        raise ValueError(f"Ablation metric index must contain a metrics list: {index_path}")
    requested = {_parse_ablation_selection(value) for value in selection_values}
    matched: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    for entry in index["metrics"]:
        if not isinstance(entry, dict):
            raise ValueError(f"Ablation metric index entries must be objects: {index_path}")
        method = entry.get("method")
        variant = entry.get("variant")
        metrics_path = entry.get("metrics_path")
        if (
            not isinstance(method, str)
            or not method
            or not isinstance(variant, str)
            or not variant
            or not isinstance(metrics_path, str)
            or not metrics_path
        ):
            raise ValueError(f"Ablation metric index entry requires method, variant, and metrics_path: {entry}")
        selection = (method, variant)
        if requested and selection not in requested:
            continue
        matched.add(selection)
        metric_rows = read_csv(metrics_path)
        if len(metric_rows) != 1:
            raise ValueError(f"Expected one metric row for ablation method={method} variant={variant}: {metrics_path}")
        metric_row = metric_rows[0]
        rows.append(
            {
                "Method": method,
                "Variant": variant,
                **{column: metric_row[column] for column in ABLATION_RESULT_COLUMNS[2:]},
            }
        )
    missing = sorted(requested - matched)
    if missing:
        values = ", ".join(f"{method}={variant}" for method, variant in missing)
        raise ValueError(f"Ablation selections are missing from the metric index: {values}")
    return rows


def _parse_ablation_selection(value: str) -> tuple[str, str]:
    method, separator, variant = value.partition("=")
    if separator != "=" or not method or not variant:
        raise ValueError(f"Ablation selection must use method=variant format: {value!r}")
    return method, variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 per-method metric CSVs into final tables.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_main", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_efficiency", required=True)
    parser.add_argument("--ablation_index")
    parser.add_argument("--output_ablation")
    parser.add_argument("--ablation_selection", action="append", default=[])
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AggregateTablesArgs:
    namespace = build_parser().parse_args(argv)
    if (namespace.ablation_index is None) != (namespace.output_ablation is None):
        raise ValueError("--ablation_index and --output_ablation must be provided together.")
    return AggregateTablesArgs(
        input_dir=namespace.input_dir,
        output_main=namespace.output_main,
        output_path=namespace.output_path,
        output_efficiency=namespace.output_efficiency,
        ablation_index=namespace.ablation_index,
        output_ablation=namespace.output_ablation,
        ablation_selections=tuple(namespace.ablation_selection),
    )


if __name__ == "__main__":
    raise SystemExit(main())
