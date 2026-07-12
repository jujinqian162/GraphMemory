from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.datasets.selection import (
    evidence_labels_for_dataset,
    text_ranking_requests_for_dataset,
)
from graph_memory.io import read_json, write_json
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import PairStageConfig
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.training_pairs import build_train_pairs
from graph_memory.training_pairs.config import NegativeSamplingConfig
from graph_memory.training_pairs.requests import TrainPairBuildTask

LOGGER = logging.getLogger("build_train_pairs")


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        PairStageConfig,
        description="Build typed training pairs from a resolved stage YAML.",
        script=Path(__file__),
    )
    config = execution.config
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    summary_path = config.outputs.pair_summary
    sampling_config = NegativeSamplingConfig(**config.sampling.model_dump())
    dense_config = _dense_config_from_config(config)

    with stage_lifecycle(execution.invocation) as observations:
        task_inputs = cast(list[Mapping[str, object]], read_json(config.tasks))
        labels = cast(list[object], read_json(config.labels))
        graphs = cast(list[MemoryGraph], read_json(config.graphs))
        result = build_train_pairs(
            _train_pair_tasks(config, task_inputs, labels, graphs),
            sampling_config,
            dense_config=dense_config,
        )
        write_json(config.outputs.pairs, result.pairs)
        write_json(summary_path, result.summary)

        observations.count("task_inputs", len(task_inputs))
        observations.count("labels", len(labels))
        observations.count("graphs", len(graphs))
        observations.count("pairs", len(result.pairs))
        observations.count("positive_count", result.summary["positive_count"])
        observations.count(
            "negative_count_by_type", result.summary["negative_count_by_type"]
        )
        observations.timing("total_seconds", time.perf_counter() - start_time)
        LOGGER.info("wrote train pairs: %s", config.outputs.pairs)
        LOGGER.info("wrote train pair summary: %s", summary_path)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


def _train_pair_tasks(
    config: PairStageConfig,
    task_inputs: list[Mapping[str, object]],
    labels: list[object],
    graphs: list[MemoryGraph],
) -> list[TrainPairBuildTask]:
    text_requests = {
        request.task_id: request
        for request in text_ranking_requests_for_dataset(config.dataset, task_inputs)
    }
    labels_by_task_id = {
        label.task_id: label
        for label in evidence_labels_for_dataset(config.dataset, labels)
    }
    graphs_by_task_id = {graph["task_id"]: graph for graph in graphs}
    tasks: list[TrainPairBuildTask] = []
    for record in task_inputs:
        task_id = str(record["task_id"])
        tasks.append(
            TrainPairBuildTask(
                text_request=text_requests[task_id],
                label=labels_by_task_id[task_id],
                graph=graphs_by_task_id[task_id],
            )
        )
    return tasks


def _dense_config_from_config(config: PairStageConfig) -> DenseConfig | None:
    if config.sampling.hard_dense_per_positive <= 0:
        return None
    encoder = config.hard_dense_encoder
    return DenseConfig(
        model_name=encoder.model_name,
        query_prefix=encoder.query_prefix,
        passage_prefix=encoder.passage_prefix,
        batch_size=encoder.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
