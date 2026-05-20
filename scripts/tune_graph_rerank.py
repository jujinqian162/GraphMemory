from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.tuning import graph_rerank_grid, tune_graph_rerank
from graph_memory.validation import validate_graphs, validate_memory_task_inputs, validate_memory_task_labels

LOGGER = logging.getLogger("tune_graph_rerank")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_config_path = Path(args.output_config)
    summary_path = output_config_path.with_name(f"{output_config_path.stem}.run_summary.json")
    candidates_path = output_config_path.with_name(f"{output_config_path.stem}.candidates.json")
    effective_config = {
        "method": args.method,
        "encoder_model": args.encoder_model,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "top_k": args.top_k,
    }
    inputs = {"tasks": args.tasks, "labels": args.labels, "graphs": args.graphs}
    outputs = {
        "selected_config": args.output_config,
        "candidate_rows": str(candidates_path),
        "run_summary": str(summary_path),
    }

    try:
        task_inputs = read_json(args.tasks)
        labels = read_json(args.labels)
        graphs = read_json(args.graphs)
        validate_memory_task_inputs(task_inputs)
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_memory_task_labels(labels, inputs_by_task_id)
        validate_graphs(graphs, inputs_by_task_id)

        grid = graph_rerank_grid()
        selected_config, candidate_rows = tune_graph_rerank(
            method=args.method,
            task_inputs=task_inputs,
            labels=labels,
            graphs=graphs,
            grid=grid,
            encoder_model=args.encoder_model,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            top_k=args.top_k,
        )
        write_json(args.output_config, selected_config)
        write_json(candidates_path, candidate_rows)

        summary = build_run_summary(
            script="tune_graph_rerank.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={"tasks": len(task_inputs), "grid_size": len(grid), "candidate_rows": len(candidate_rows)},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("selected config: %s", selected_config)
        LOGGER.info("wrote selected config: %s", args.output_config)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="tune_graph_rerank.py",
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
    parser = argparse.ArgumentParser(description="Tune graph rerank parameters on dev labels.")
    parser.add_argument("--method", required=True, choices=["bm25_graph_rerank", "dense_graph_rerank"])
    parser.add_argument("--tasks", required=True, help="Path to dev *_memory_tasks.input.json.")
    parser.add_argument("--labels", required=True, help="Path to dev *_memory_tasks.labels.json.")
    parser.add_argument("--graphs", required=True, help="Path to dev *_graphs.json.")
    parser.add_argument("--output_config", required=True, help="Path to write selected graph rerank config JSON.")
    parser.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--top_k", type=int, default=10)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
