from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.retrieval import run_retrieval
from graph_memory.retrieval_registry import get_supported_methods
from graph_memory.validation import (
    validate_memory_task_inputs,
    validate_ranked_results,
)

LOGGER = logging.getLogger("run_retrieval")


@dataclass(frozen=True)
class RunRetrievalArgs:
    method: str
    tasks: str
    graphs: str | None
    output: str
    top_k: int
    encoder_model: str
    query_prefix: str
    passage_prefix: str
    graph_config: str | None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}.run_summary.json")
    graph_config = read_json(args.graph_config) if args.graph_config is not None else None
    effective_config = {
        "method": args.method,
        "top_k": args.top_k,
        "encoder_model": args.encoder_model,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "graph_config_path": args.graph_config,
        "graph_config": graph_config,
    }
    inputs = {"tasks": args.tasks}
    if args.graphs is not None:
        inputs["graphs"] = args.graphs
    outputs = {"predictions": args.output, "run_summary": str(summary_path)}

    try:
        task_inputs = read_json(args.tasks)
        validate_memory_task_inputs(task_inputs)
        graphs = read_json(args.graphs) if args.graphs is not None else []
        predictions = run_retrieval(
            method=args.method,
            task_inputs=task_inputs,
            graphs=graphs,
            top_k=args.top_k,
            encoder_model=args.encoder_model,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            graph_config=graph_config,
        )
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_ranked_results(predictions, inputs_by_task_id)
        write_json(args.output, predictions)

        avg_latency = (
            sum(prediction["latency_ms"] for prediction in predictions) / len(predictions)
            if predictions
            else 0.0
        )
        summary = build_run_summary(
            script="run_retrieval.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={"tasks": len(task_inputs), "predictions": len(predictions)},
            timings={"total_seconds": time.perf_counter() - start_time, "avg_latency_ms": avg_latency},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("method=%s tasks=%s top_k=%s", args.method, len(task_inputs), args.top_k)
        LOGGER.info("wrote predictions: %s", args.output)
        LOGGER.info("wrote run summary: %s", summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="run_retrieval.py",
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
    parser = argparse.ArgumentParser(description="Run Phase 1 retrieval methods.")
    parser.add_argument("--method", required=True, choices=get_supported_methods())
    parser.add_argument("--tasks", required=True, help="Path to *_memory_tasks.input.json.")
    parser.add_argument("--graphs", default=None, help="Path to *_graphs.json. Required for graph rerank methods.")
    parser.add_argument("--output", required=True, help="Path to write ranked result JSON.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--graph_config", default=None, help="Path to graph rerank config JSON.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> RunRetrievalArgs:
    namespace = build_parser().parse_args(argv)
    return RunRetrievalArgs(
        method=namespace.method,
        tasks=namespace.tasks,
        graphs=namespace.graphs,
        output=namespace.output,
        top_k=namespace.top_k,
        encoder_model=namespace.encoder_model,
        query_prefix=namespace.query_prefix,
        passage_prefix=namespace.passage_prefix,
        graph_config=namespace.graph_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
