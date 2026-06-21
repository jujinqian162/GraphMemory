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

from graph_memory.datasets.selection import DatasetId, graph_build_requests_for_dataset, validate_ranking_records_for_dataset
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.builder import build_graphs
from graph_memory.graphs.statistics import graph_statistics
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.validation import validate_graphs

LOGGER = logging.getLogger("build_graphs")


@dataclass(frozen=True)
class BuildGraphsArgs:
    input: str
    output: str
    dataset: DatasetId
    max_query_overlap: int
    max_entity_neighbors: int
    max_bridge_edges: int
    use_spacy: bool


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
        "dataset": args.dataset,
        "max_query_overlap": config.max_query_overlap,
        "max_entity_neighbors": config.max_entity_neighbors,
        "max_bridge_edges": config.max_bridge_edges,
        "use_spacy": config.use_spacy,
    }
    inputs = {"tasks": args.input}
    outputs = {"graphs": args.output, "graph_stats": str(stats_path), "run_summary": str(summary_path)}

    try:
        ranking_records = read_json(args.input)
        validate_ranking_records_for_dataset(args.dataset, ranking_records)
        LOGGER.info("read %s ranking records: %s", args.dataset, len(ranking_records))

        graph_requests = graph_build_requests_for_dataset(args.dataset, ranking_records)
        graphs = build_graphs(graph_requests, config)
        validate_graphs(graphs, graph_requests)
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
                "ranking_records": len(ranking_records),
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
    parser = argparse.ArgumentParser(description="Build typed graphs from dataset ranking records.")
    parser.add_argument("--dataset", choices=("hotpotqa", "twowiki"), default="hotpotqa")
    parser.add_argument("--input", required=True, help="Path to dataset ranking record JSON.")
    parser.add_argument("--output", required=True, help="Path to write *_graphs.json.")
    parser.add_argument("--max_query_overlap", type=int, default=20)
    parser.add_argument("--max_entity_neighbors", type=int, default=10)
    parser.add_argument("--max_bridge_edges", type=int, default=50)
    parser.add_argument("--use_spacy", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> BuildGraphsArgs:
    namespace = build_parser().parse_args(argv)
    return BuildGraphsArgs(
        input=namespace.input,
        output=namespace.output,
        dataset=cast(DatasetId, namespace.dataset),
        max_query_overlap=namespace.max_query_overlap,
        max_entity_neighbors=namespace.max_entity_neighbors,
        max_bridge_edges=namespace.max_bridge_edges,
        use_spacy=namespace.use_spacy,
    )


if __name__ == "__main__":
    raise SystemExit(main())
