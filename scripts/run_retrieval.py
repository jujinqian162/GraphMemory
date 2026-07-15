from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import TypeAdapter

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.datasets.selection import (
    text_ranking_requests_for_dataset,
    validate_ranking_records_for_dataset,
)
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import (
    RetrieveStageConfig,
    RgcnRetrieveStageConfig,
)
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.io import read_json, write_json
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.stages.retrieve import run_retrieve_stage
from graph_memory.validation import validate_ranked_results

LOGGER = logging.getLogger("run_retrieval")


def main(argv: Sequence[str] | None = None) -> int:
    execution = load_stage_execution(
        None if argv is None else list(argv),
        TypeAdapter(RetrieveStageConfig),
        description="Run retrieval from a resolved stage YAML.",
        script=Path(__file__),
    )
    config = execution.config
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s"
    )

    start_time = time.perf_counter()
    with stage_lifecycle(execution.invocation) as observations:
        task_inputs = read_json(config.tasks)
        validate_ranking_records_for_dataset(config.dataset, task_inputs)
        text_requests = _text_requests(config, cast(list[object], task_inputs))
        evidence_graphs = _read_evidence_graphs(config)
        result = run_retrieve_stage(
            config,
            task_inputs=task_inputs,
            evidence_graphs=evidence_graphs,
        )
        predictions = result.predictions
        validate_ranked_results(predictions, text_requests)
        write_json(config.output, predictions)

        avg_latency = (
            sum(prediction["latency_ms"] for prediction in predictions)
            / len(predictions)
            if predictions
            else 0.0
        )
        observations.count("tasks", len(task_inputs))
        observations.count("predictions", len(predictions))
        observations.count("provenance_method", result.provenance.method.value)
        if isinstance(config, RgcnRetrieveStageConfig) and predictions:
            observations.count("retrieval_device", result.provenance.device)
        observations.timing("total_seconds", time.perf_counter() - start_time)
        observations.timing("avg_latency_ms", avg_latency)
        LOGGER.info(
            "method=%s tasks=%s top_k=%s",
            config.method,
            len(task_inputs),
            config.top_k,
        )
        LOGGER.info("wrote predictions: %s", config.output)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


def _text_requests(
    config: RetrieveStageConfig,
    records: Sequence[object],
) -> list[TextRankingRequest]:
    return text_ranking_requests_for_dataset(config.dataset, records)


def _read_evidence_graphs(
    config: RetrieveStageConfig,
) -> list[EvidenceGraph]:
    if not isinstance(config, RgcnRetrieveStageConfig):
        return []
    value = read_json(config.evidence_graphs)
    if not isinstance(value, list):
        raise ValueError(
            f"EvidenceGraph artifact must be a JSON list: {config.evidence_graphs}"
        )
    return cast(list[EvidenceGraph], value)


if __name__ == "__main__":
    raise SystemExit(main())
