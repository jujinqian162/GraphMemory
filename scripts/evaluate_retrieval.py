from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.evaluation.failure_cases import build_failure_cases
from graph_memory.evaluation.service import evaluate_results
from graph_memory.evaluation.tables import WIDE_METRIC_COLUMNS
from graph_memory.io import read_json, write_csv, write_jsonl
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.validation import validate_metric_rows

LOGGER = logging.getLogger("evaluate_retrieval")


@dataclass(frozen=True)
class EvaluateRetrievalArgs:
    pred: str
    labels: str | None
    gold: str | None
    graphs: str
    output: str
    failure_cases_output: str | None
    failure_case_limit: int


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}.run_summary.json")
    label_path = args.labels or args.gold
    if label_path is None:
        parser.error("--labels is required; --gold is accepted as a compatibility alias.")
    effective_config = {
        "failure_case_limit": args.failure_case_limit,
        "failure_cases_output": args.failure_cases_output,
    }
    inputs = {"predictions": args.pred, "labels": label_path, "graphs": args.graphs}
    outputs = {"metrics": args.output, "run_summary": str(summary_path)}
    if args.failure_cases_output is not None:
        outputs["failure_cases"] = args.failure_cases_output

    try:
        predictions = read_json(args.pred)
        labels = read_json(label_path)
        graphs = read_json(args.graphs)
        metric_rows = evaluate_results(predictions, labels, graphs)
        validate_metric_rows(metric_rows)
        write_csv(args.output, metric_rows, WIDE_METRIC_COLUMNS)

        failure_cases = build_failure_cases(
            predictions,
            labels,
            graphs,
            top_k=10,
            limit=args.failure_case_limit,
        )
        if args.failure_cases_output is not None:
            write_jsonl(args.failure_cases_output, failure_cases)

        summary = build_run_summary(
            script="evaluate_retrieval.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={"predictions": len(predictions), "metric_rows": len(metric_rows), "failure_cases": len(failure_cases)},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote metrics: %s", args.output)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 ranked retrieval results.")
    parser.add_argument("--pred", required=True, help="Path to ranked result JSON.")
    parser.add_argument("--labels", default=None, help="Path to *_memory_tasks.labels.json.")
    parser.add_argument("--gold", default=None, help="Compatibility alias for --labels.")
    parser.add_argument("--graphs", required=True, help="Path to *_graphs.json.")
    parser.add_argument("--output", required=True, help="Path to write per-method metric CSV.")
    parser.add_argument("--failure_cases_output", default=None, help="Optional JSONL failure-case debug output.")
    parser.add_argument("--failure_case_limit", type=int, default=0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> EvaluateRetrievalArgs:
    namespace = build_parser().parse_args(argv)
    return EvaluateRetrievalArgs(
        pred=namespace.pred,
        labels=namespace.labels,
        gold=namespace.gold,
        graphs=namespace.graphs,
        output=namespace.output,
        failure_cases_output=namespace.failure_cases_output,
        failure_case_limit=namespace.failure_case_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
