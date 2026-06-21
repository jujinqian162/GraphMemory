from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.evaluation.tables import WIDE_METRIC_COLUMNS
from graph_memory.io import read_json, write_csv, write_jsonl
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.stages.evaluate import run_evaluate_stage
from graph_memory.validation import validate_metric_rows

LOGGER = logging.getLogger("evaluate_retrieval")


def main(argv: Sequence[str] | None = None) -> int:
    config = CONFIG_LOADER.load(Registry.configs.EVALUATE, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_path = config.io.output
    summary_path = output_path.with_name(f"{output_path.stem}.run_summary.json")
    effective_config = {
        "dataset": config.dataset,
        "failure_case_limit": config.failure_case_limit,
        "failure_cases_output": str(config.io.failure_cases_output) if config.io.failure_cases_output is not None else None,
    }
    inputs = {
        "predictions": str(config.io.predictions),
        "labels": str(config.io.labels),
        "graphs": str(config.io.graphs),
    }
    outputs = {"metrics": str(config.io.output), "run_summary": str(summary_path)}
    if config.io.failure_cases_output is not None:
        outputs["failure_cases"] = str(config.io.failure_cases_output)

    try:
        predictions = read_json(config.io.predictions)
        labels = read_json(config.io.labels)
        graphs = read_json(config.io.graphs)
        result = run_evaluate_stage(
            config,
            predictions=predictions,
            labels=labels,
            graphs=graphs,
        )
        validate_metric_rows(result.metric_rows)
        write_csv(config.io.output, result.metric_rows, WIDE_METRIC_COLUMNS)

        if config.io.failure_cases_output is not None:
            write_jsonl(config.io.failure_cases_output, result.failure_cases)

        summary = build_run_summary(
            script="evaluate_retrieval.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={
                "predictions": len(predictions),
                "metric_rows": len(result.metric_rows),
                "failure_cases": len(result.failure_cases),
            },
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote metrics: %s", config.io.output)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="evaluate_retrieval.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config=effective_config,
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


if __name__ == "__main__":
    raise SystemExit(main())
