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

from graph_memory.evaluation import (
    EFFICIENCY_RESULT_COLUMNS,
    MAIN_RESULT_COLUMNS,
    PATH_RESULT_COLUMNS,
    split_metric_tables,
)
from graph_memory.io import read_csv, write_csv
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.types import MetricRow

LOGGER = logging.getLogger("aggregate_tables")


@dataclass(frozen=True)
class AggregateTablesArgs:
    input_dir: str
    output_main: str
    output_path: str
    output_efficiency: str


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_main = Path(args.output_main)
    summary_path = output_main.with_name("aggregate_tables.run_summary.json")
    output_paths = {Path(args.output_main), Path(args.output_path), Path(args.output_efficiency)}
    inputs = {"input_dir": args.input_dir}
    outputs = {
        "main": args.output_main,
        "path": args.output_path,
        "efficiency": args.output_efficiency,
        "run_summary": str(summary_path),
    }

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 per-method metric CSVs into final tables.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_main", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_efficiency", required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AggregateTablesArgs:
    namespace = build_parser().parse_args(argv)
    return AggregateTablesArgs(
        input_dir=namespace.input_dir,
        output_main=namespace.output_main,
        output_path=namespace.output_path,
        output_efficiency=namespace.output_efficiency,
    )


if __name__ == "__main__":
    raise SystemExit(main())
