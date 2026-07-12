from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.datasets.selection import (
    graph_build_requests_for_dataset,
    validate_ranking_records_for_dataset,
)
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.builder import build_graphs
from graph_memory.graphs.statistics import graph_statistics
from graph_memory.io import read_json, write_json
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import GraphStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.validation import validate_graphs

LOGGER = logging.getLogger("build_graphs")


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        GraphStageConfig,
        description="Build typed graphs from a resolved stage YAML.",
        script=Path(__file__),
    )
    stage_config = execution.config
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    output_path = stage_config.output
    stats_path = output_path.with_name(f"{output_path.stem}.stats.json")
    config = GraphBuildConfig(
        max_query_overlap=stage_config.graph.max_query_overlap,
        max_entity_neighbors=stage_config.graph.max_entity_neighbors,
        max_bridge_edges=stage_config.graph.max_bridge_edges,
        use_spacy=stage_config.graph.use_spacy,
    )
    effective_config = {
        "dataset": stage_config.dataset,
        "max_query_overlap": config.max_query_overlap,
        "max_entity_neighbors": config.max_entity_neighbors,
        "max_bridge_edges": config.max_bridge_edges,
        "use_spacy": config.use_spacy,
    }
    with stage_lifecycle(execution.invocation) as observations:
        ranking_records = read_json(stage_config.tasks)
        validate_ranking_records_for_dataset(stage_config.dataset, ranking_records)
        LOGGER.info(
            "read %s ranking records: %s", stage_config.dataset, len(ranking_records)
        )

        graph_requests = graph_build_requests_for_dataset(
            stage_config.dataset, ranking_records
        )
        graphs = build_graphs(graph_requests, config)
        validate_graphs(graphs, graph_requests)
        stats = graph_statistics(graphs, graph_config=effective_config)
        write_json(stage_config.output, graphs)
        write_json(stats_path, stats)

        observations.count("ranking_records", len(ranking_records))
        observations.count("graphs", len(graphs))
        observations.count("avg_nodes", stats["avg_nodes"])
        observations.count("avg_edges", stats["avg_edges"])
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("wrote graphs: %s", stage_config.output)
        LOGGER.info("wrote graph stats: %s", stats_path)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
