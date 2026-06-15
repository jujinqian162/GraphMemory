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

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonObject, JsonValue
from graph_memory.io import read_json, write_json
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    GraphRerankRetrievalSettings,
    MemoryStreamRetrievalSettings,
    RetrievalProvenance,
)
from graph_memory.registry.stage_configs import RetrieveStageConfig
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.graph_rerank.config import (
    GraphRerankConfig,
    ensure_graph_rerank_config,
)
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    memory_stream_scoring_config_record,
    parse_memory_stream_scoring_config,
)
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
    selected_config = _read_selected_config(config)
    effective_config = _effective_config(config, selected_config, provenance=None)
    inputs = {"tasks": str(config.io.tasks)}
    if config.io.graphs is not None:
        inputs["graphs"] = str(config.io.graphs)
    if config.io.importance is not None:
        inputs["importance"] = str(config.io.importance)
    if config.io.selected_config is not None:
        inputs["selected_config"] = str(config.io.selected_config)
    outputs = {"predictions": str(config.io.output), "run_summary": str(summary_path)}

    try:
        task_inputs = read_json(config.io.tasks)
        validate_memory_task_inputs(task_inputs)
        importance_artifact, importance_sha256 = _load_memory_stream_importance_if_required(config)
        graphs = read_json(config.io.graphs) if config.io.graphs is not None else []
        result = run_retrieve_stage(
            config,
            task_inputs=task_inputs,
            graphs=graphs,
            selected_config=selected_config,
            importance_artifact=importance_artifact,
            importance_sha256=importance_sha256,
        )
        predictions = result.predictions
        effective_config = _effective_config(config, selected_config, provenance=result.provenance)
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


SelectedConfig = GraphRerankConfig | MemoryStreamScoringConfig


def _read_selected_config(config: RetrieveStageConfig) -> SelectedConfig | None:
    path = config.io.selected_config
    if path is None:
        return None
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"Selected config must be a JSON object: {path}")
    if isinstance(config.job, GraphRerankRetrievalSettings):
        return ensure_graph_rerank_config(value)
    if isinstance(config.job, MemoryStreamRetrievalSettings):
        return parse_memory_stream_scoring_config(value)
    raise ValueError(
        f"Retrieval method={config.job.method.value} does not accept selected_config."
    )


def _load_memory_stream_importance_if_required(
    config: RetrieveStageConfig,
) -> tuple[ImportanceArtifact | None, str | None]:
    if not isinstance(config.job, MemoryStreamRetrievalSettings):
        return None, None
    importance_path = config.io.importance
    if importance_path is None:
        raise ValueError("Memory Stream retrieval requires an importance artifact path.")
    if not importance_path.is_file():
        raise ValueError(f"Memory Stream importance artifact not found: {importance_path}")
    raw_bytes = importance_path.read_bytes()
    artifact = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError(f"Memory Stream importance artifact must be a JSON object: {importance_path}")
    return cast(ImportanceArtifact, cast(object, artifact)), hashlib.sha256(raw_bytes).hexdigest()


def _effective_config(
    config: RetrieveStageConfig,
    selected_config: SelectedConfig | None,
    *,
    provenance: RetrievalProvenance | None,
) -> JsonObject:
    return {
        "method": config.job.method.value,
        "top_k": config.job.top_k,
        "job": cast(JsonObject, CONFIG_LOADER.to_json(config.job)),
        "selected_config_path": (
            str(config.io.selected_config)
            if config.io.selected_config is not None
            else None
        ),
        "importance_path": str(config.io.importance) if config.io.importance is not None else None,
        "selected_config": _selected_config_json(selected_config),
        "provenance": None if provenance is None else cast(JsonObject, CONFIG_LOADER.to_json(provenance)),
    }


def _selected_config_json(selected_config: SelectedConfig | None) -> JsonValue:
    if selected_config is None:
        return None
    if isinstance(selected_config, MemoryStreamScoringConfig):
        return cast(JsonObject, memory_stream_scoring_config_record(selected_config))
    return cast(JsonObject, CONFIG_LOADER.to_json(selected_config))


if __name__ == "__main__":
    raise SystemExit(main())
