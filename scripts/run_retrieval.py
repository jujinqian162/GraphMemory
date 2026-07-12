from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.datasets.selection import (
    text_ranking_requests_for_dataset,
    validate_ranking_records_for_dataset,
)
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.experiment.stage_cli import load_stage_execution
from graph_memory.experiment.stage_models import (
    Bm25GraphRerankRetrieveStageConfig,
    DenseGraphRerankRetrieveStageConfig,
    MemoryStreamRetrieveStageConfig,
    RetrieveStageConfig,
    RgcnRetrieveStageConfig,
)
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.io import read_json, write_json
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.graph_rerank.config import (
    GraphRerankConfig,
    ensure_graph_rerank_config,
)
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    parse_memory_stream_scoring_config,
)
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.stages.retrieve import run_retrieve_stage
from graph_memory.validation import validate_ranked_results
from pydantic import TypeAdapter

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
    selected_config = _read_selected_config(config)

    with stage_lifecycle(execution.invocation) as observations:
        task_inputs = read_json(config.tasks)
        validate_ranking_records_for_dataset(config.dataset, task_inputs)
        text_requests = _text_requests(config, cast(list[object], task_inputs))
        importance_artifact, importance_sha256 = (
            _load_memory_stream_importance_if_required(config)
        )
        graphs = _read_graphs(config)
        result = run_retrieve_stage(
            config,
            task_inputs=task_inputs,
            graphs=graphs,
            selected_config=selected_config,
            importance_artifact=importance_artifact,
            importance_sha256=importance_sha256,
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
        observations.timing("total_seconds", time.perf_counter() - start_time)
        observations.timing("avg_latency_ms", avg_latency)
        LOGGER.info(
            "method=%s tasks=%s top_k=%s", config.method, len(task_inputs), config.top_k
        )
        LOGGER.info("wrote predictions: %s", config.output)
    LOGGER.info("wrote run summary: %s", execution.invocation.summary_path)
    return 0


def _text_requests(
    config: RetrieveStageConfig, records: Sequence[object]
) -> list[TextRankingRequest]:
    return text_ranking_requests_for_dataset(config.dataset, records)


SelectedConfig = GraphRerankConfig | MemoryStreamScoringConfig


def _read_selected_config(config: RetrieveStageConfig) -> SelectedConfig | None:
    if isinstance(
        config,
        (Bm25GraphRerankRetrieveStageConfig, DenseGraphRerankRetrieveStageConfig),
    ):
        value = read_json(config.selected_config)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Selected config must be a JSON object: {config.selected_config}"
            )
        return ensure_graph_rerank_config(value)
    if isinstance(config, MemoryStreamRetrieveStageConfig):
        value = read_json(config.selected_config)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Selected config must be a JSON object: {config.selected_config}"
            )
        return parse_memory_stream_scoring_config(value)
    return None


def _read_graphs(config: RetrieveStageConfig) -> list[MemoryGraph]:
    if not isinstance(
        config,
        (
            Bm25GraphRerankRetrieveStageConfig,
            DenseGraphRerankRetrieveStageConfig,
            RgcnRetrieveStageConfig,
        ),
    ):
        return []
    value = read_json(config.graphs)
    if not isinstance(value, list):
        raise ValueError(f"Graph artifact must be a JSON list: {config.graphs}")
    return cast(list[MemoryGraph], value)


def _load_memory_stream_importance_if_required(
    config: RetrieveStageConfig,
) -> tuple[ImportanceArtifact | None, str | None]:
    if not isinstance(config, MemoryStreamRetrieveStageConfig):
        return None, None
    importance_path = config.importance
    if not importance_path.is_file():
        raise ValueError(
            f"Memory Stream importance artifact not found: {importance_path}"
        )
    raw_bytes = importance_path.read_bytes()
    artifact = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError(
            f"Memory Stream importance artifact must be a JSON object: {importance_path}"
        )
    return cast(ImportanceArtifact, cast(object, artifact)), hashlib.sha256(
        raw_bytes
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
