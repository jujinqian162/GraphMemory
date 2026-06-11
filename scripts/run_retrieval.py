from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonObject, JsonValue
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    CheckpointGraphRetrievalSettings,
    DenseRetrievalSettings,
    GraphRerankRetrievalSettings,
    RetrievalJobSettings,
)
from graph_memory.registry.stage_configs import RetrieveStageConfig
from graph_memory.stages.retrieve import run_retrieve_stage
from graph_memory.validation import (
    validate_memory_task_inputs,
    validate_ranked_results,
)

LOGGER = logging.getLogger("run_retrieval")


def main(argv: Sequence[str] | None = None) -> int:
    config = CONFIG_LOADER.load(Registry.configs.RETRIEVE, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    summary_path = config.io.summary
    graph_config = _read_graph_config(config.io.graph_config)
    effective_config = _effective_config(config, graph_config)
    inputs = {"tasks": str(config.io.tasks)}
    if config.io.graphs is not None:
        inputs["graphs"] = str(config.io.graphs)
    outputs = {"predictions": str(config.io.output), "run_summary": str(summary_path)}

    try:
        task_inputs = read_json(config.io.tasks)
        validate_memory_task_inputs(task_inputs)
        graphs = read_json(config.io.graphs) if config.io.graphs is not None else []
        result = run_retrieve_stage(
            config,
            task_inputs=task_inputs,
            graphs=graphs,
            graph_config=graph_config,
        )
        predictions = result.predictions
        inputs_by_task_id = {task_input["task_id"]: task_input for task_input in task_inputs}
        validate_ranked_results(predictions, inputs_by_task_id)
        write_json(config.io.output, predictions)

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
        LOGGER.info("method=%s tasks=%s top_k=%s", config.job.method.value, len(task_inputs), config.job.top_k)
        LOGGER.info("wrote predictions: %s", config.io.output)
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


def _read_graph_config(path: Path | None) -> JsonObject | None:
    if path is None:
        return None
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"Graph rerank config must be a JSON object: {path}")
    return cast(JsonObject, value)


def _effective_config(config: RetrieveStageConfig, graph_config: JsonValue) -> JsonObject:
    encoder = _encoder_settings(config)
    return {
        "method": config.job.method.value,
        "top_k": config.job.top_k,
        "encoder_model": encoder["model_name"],
        "query_prefix": encoder["query_prefix"],
        "passage_prefix": encoder["passage_prefix"],
        "graph_config_path": str(config.io.graph_config) if config.io.graph_config is not None else None,
        "graph_config": graph_config,
        "checkpoint": _checkpoint_path(config.job),
        "device": _device(config.job),
    }


def _encoder_settings(config: RetrieveStageConfig) -> dict[str, str]:
    job = config.job
    if isinstance(job, DenseRetrievalSettings):
        return {
            "model_name": job.encoder.model_name,
            "query_prefix": job.encoder.query_prefix,
            "passage_prefix": job.encoder.passage_prefix,
        }
    if isinstance(job, GraphRerankRetrievalSettings) and job.seed.encoder is not None:
        return {
            "model_name": job.seed.encoder.model_name,
            "query_prefix": job.seed.encoder.query_prefix,
            "passage_prefix": job.seed.encoder.passage_prefix,
        }
    return {
        "model_name": config.io.encoder_model,
        "query_prefix": config.io.query_prefix,
        "passage_prefix": config.io.passage_prefix,
    }


def _checkpoint_path(job: RetrievalJobSettings) -> str | None:
    if isinstance(job, CheckpointGraphRetrievalSettings):
        return str(job.checkpoint)
    return None


def _device(job: RetrievalJobSettings) -> str:
    if isinstance(job, CheckpointGraphRetrievalSettings):
        return job.device
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
