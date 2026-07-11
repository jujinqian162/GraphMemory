from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.datasets.selection import (
    evidence_labels_for_dataset,
    text_ranking_requests_for_dataset,
    validate_label_records_for_dataset,
    validate_ranking_records_for_dataset,
)
from graph_memory.io import read_json, write_json
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import GraphRerankTuneStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.retrieval.tuning import (
    graph_rerank_grid_from_record,
    tune_graph_rerank,
)
from graph_memory.validation import validate_graphs

LOGGER = logging.getLogger("tune_graph_rerank")


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        GraphRerankTuneStageConfig,
        description="Tune graph reranking from a resolved stage YAML.",
        script=Path(__file__),
    )
    config = execution.config
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        task_inputs = cast(list[object], read_json(config.tasks))
        labels = cast(list[object], read_json(config.labels))
        graphs = read_json(config.graphs)
        validate_ranking_records_for_dataset(config.dataset, task_inputs)
        inputs_by_task_id = {
            str(task_input["task_id"]): task_input
            for task_input in cast(list[Mapping[str, object]], task_inputs)
        }
        validate_label_records_for_dataset(config.dataset, labels, inputs_by_task_id)
        ranking_requests = text_ranking_requests_for_dataset(
            config.dataset, task_inputs
        )
        evidence_labels = evidence_labels_for_dataset(config.dataset, labels)
        validate_graphs(graphs, ranking_requests)

        grid = graph_rerank_grid_from_record(config.search_space.model_dump())
        dense_runtime = None
        if config.seed_encoder is not None:
            encoder = config.seed_encoder
            dense_runtime = DenseRuntime(
                config=DenseConfig(
                    model_name=encoder.model_name,
                    query_prefix=encoder.query_prefix,
                    passage_prefix=encoder.passage_prefix,
                    batch_size=encoder.batch_size,
                )
            )
        selected_config, candidate_rows = tune_graph_rerank(
            method=config.method,
            ranking_requests=ranking_requests,
            labels=evidence_labels,
            graphs=graphs,
            grid=grid,
            top_k=config.top_k,
            dense_runtime=dense_runtime,
        )
        write_json(config.selected_config, selected_config)
        write_json(config.candidates, candidate_rows)

        observations.count("tasks", len(task_inputs))
        observations.count("grid_size", len(grid))
        observations.count("candidate_rows", len(candidate_rows))
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("selected config: %s", selected_config)
        LOGGER.info("wrote selected config: %s", config.selected_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
