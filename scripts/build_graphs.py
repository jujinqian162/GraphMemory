from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.graphs import build_graphs
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, graph_statistics, now_iso, write_run_summary
from graph_memory.types import GraphBuildConfig
from graph_memory.validation import validate_graphs, validate_memory_task_inputs

LOGGER = logging.getLogger("build_graphs")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}.run_summary.json")
    stats_path = output_path.with_name(f"{output_path.stem}.stats.json")
    config = GraphBuildConfig(
        max_query_overlap=args.max_query_overlap,
        max_entity_neighbors=args.max_entity_neighbors,
        max_bridge_edges=args.max_bridge_edges,
        use_spacy=args.use_spacy,
    )
    effective_config = {
        "max_query_overlap": config.max_query_overlap,
        "max_entity_neighbors": config.max_entity_neighbors,
        "max_bridge_edges": config.max_bridge_edges,
        "use_spacy": config.use_spacy,
    }
    inputs = {"tasks": args.input}
    outputs = {"graphs": args.output, "graph_stats": str(stats_path), "run_summary": str(summary_path)}

    try:
        task_inputs = read_json(args.input)
        validate_memory_task_inputs(task_inputs)
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        LOGGER.info("read task inputs: %s", len(task_inputs))

        graphs = build_graphs(task_inputs, config)
        validate_graphs(graphs, inputs_by_task_id)
        stats = graph_statistics(graphs, graph_config=effective_config)
        write_json(args.output, graphs)
        write_json(stats_path, stats)

        summary = build_run_summary(
            script="build_graphs.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs=outputs,
            counts={
                "task_inputs": len(task_inputs),
                "graphs": len(graphs),
                "avg_nodes": stats["avg_nodes"],
                "avg_edges": stats["avg_edges"],
            },
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(summary_path, summary)
        LOGGER.info("wrote graphs: %s", args.output)
        LOGGER.info("wrote graph stats: %s", stats_path)
        LOGGER.info("wrote run summary: %s", summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="build_graphs.py",
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
    parser = argparse.ArgumentParser(description="Build typed memory graphs from leakage-safe task inputs.")
    parser.add_argument("--input", required=True, help="Path to *_memory_tasks.input.json.")
    parser.add_argument("--output", required=True, help="Path to write *_graphs.json.")
    parser.add_argument("--max_query_overlap", type=int, default=20)
    parser.add_argument("--max_entity_neighbors", type=int, default=10)
    parser.add_argument("--max_bridge_edges", type=int, default=50)
    parser.add_argument("--use_spacy", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
