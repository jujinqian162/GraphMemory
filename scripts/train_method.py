from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonObject, JsonValue
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.dense_finetune.training import DenseFinetuneTrainingResult
from graph_memory.models.graph_retriever.checkpoint import save_trainable_checkpoint
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.training import TrainableTrainingResult
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import (
    DenseFinetuneTrainStageConfig,
    RgcnTrainStageConfig,
    TrainStageConfig,
)
from graph_memory.registry.training import (
    DenseFinetuneTrainPayload,
    RgcnTrainPayload,
    TrainDependencies,
    TrainPayload,
)
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.stages.train import run_train_stage

LOGGER = logging.getLogger("train_method")


def main(
    argv: Sequence[str] | None = None,
    *,
    text_embedding_provider: TextEmbeddingProvider | None = None,
    seed_signal_provider: SeedSignalProvider | None = None,
) -> int:
    config = CONFIG_LOADER.load(Registry.configs.TRAIN, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    dependencies = _injected_dependencies(text_embedding_provider, seed_signal_provider)
    started_at = now_iso()
    start_time = time.perf_counter()
    inputs = _input_paths(config)
    outputs = _output_paths(config)

    try:
        payload = _load_payload(config, dependencies=dependencies)
        training_result = run_train_stage(config, payload=payload).result
        config.io.output_dir.mkdir(parents=True, exist_ok=True)
        metric_records = _metric_records(training_result)
        write_jsonl(config.io.metrics, metric_records)
        artifact_outputs = _write_method_artifacts(config, training_result)
        summary = build_run_summary(
            script="train_method.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=_effective_config(config, training_result),
            inputs=inputs,
            outputs={**outputs, **artifact_outputs},
            counts=_result_counts(payload, training_result),
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(config.io.run_summary, summary)
        LOGGER.info("wrote train output: %s", outputs["best_checkpoint"])
        LOGGER.info("wrote metrics: %s", config.io.metrics)
        LOGGER.info("wrote run summary: %s", config.io.run_summary)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="train_method.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config={"method": config.method.value},
            inputs=inputs,
            outputs=outputs,
            counts={},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
            error=str(error),
        )
        write_run_summary(config.io.run_summary, summary)
        raise


def _injected_dependencies(
    text_embedding_provider: TextEmbeddingProvider | None,
    seed_signal_provider: SeedSignalProvider | None,
) -> TrainDependencies | None:
    if text_embedding_provider is None and seed_signal_provider is None:
        return None
    if text_embedding_provider is None or seed_signal_provider is None:
        raise ValueError("Train provider overrides require both text_embedding_provider and seed_signal_provider.")
    return TrainDependencies(
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
    )


def _load_payload(config: TrainStageConfig, *, dependencies: TrainDependencies | None) -> TrainPayload:
    if isinstance(config, RgcnTrainStageConfig):
        return RgcnTrainPayload(
            train_task_inputs=cast(list[MemoryTaskInput], read_json(config.io.train_tasks)),
            train_labels=(
                None
                if config.io.train_labels is None
                else cast(list[MemoryTaskLabels], read_json(config.io.train_labels))
            ),
            train_graphs=cast(list[MemoryGraph], read_json(config.io.train_graphs)),
            train_pairs=cast(list[TrainPairRecord], read_json(config.io.train_pairs)),
            dev_task_inputs=cast(list[MemoryTaskInput], read_json(config.io.dev_tasks)),
            dev_labels=cast(list[MemoryTaskLabels], read_json(config.io.dev_labels)),
            dev_graphs=cast(list[MemoryGraph], read_json(config.io.dev_graphs)),
            dependencies=dependencies,
        )
    if dependencies is not None:
        raise ValueError("Provider overrides are only valid for R-GCN training.")
    return DenseFinetuneTrainPayload(
        train_task_inputs=cast(list[MemoryTaskInput], read_json(config.io.train_tasks)),
        train_labels=cast(list[MemoryTaskLabels], read_json(config.io.train_labels)),
        train_pairs=cast(list[TrainPairRecord], read_json(config.io.train_pairs)),
        dev_task_inputs=cast(list[MemoryTaskInput], read_json(config.io.dev_tasks)),
        dev_labels=cast(list[MemoryTaskLabels], read_json(config.io.dev_labels)),
        output_dir=config.io.output_dir,
        model_dir=config.io.model_dir,
    )


def _write_method_artifacts(config: TrainStageConfig, result: object) -> JsonObject:
    if isinstance(config, RgcnTrainStageConfig):
        if not isinstance(result, TrainableTrainingResult):
            raise TypeError(f"R-GCN training returned {type(result).__name__}.")
        best_model = build_model_from_config(result.model_config)
        best_model.load_state_dict(result.best_model_state_dict)
        epoch_checkpoint = config.io.checkpoint_dir / f"checkpoint_epoch_{result.best_epoch}.pt"
        for path in (epoch_checkpoint, config.io.checkpoint_dir / "best.pt"):
            save_trainable_checkpoint(
                path,
                method_name=result.model_config.method_name,
                model=best_model,
                optimizer_state_dict=result.optimizer_state_dict,
                scheduler_state_dict=result.scheduler_state_dict,
                epoch=result.best_epoch,
                global_step=result.global_step,
                best_dev_metric=result.best_dev_metric,
                model_config=result.model_config,
                training_config=result.training_config,
            )
        return {"epoch_checkpoint": str(epoch_checkpoint)}
    if not isinstance(result, DenseFinetuneTrainingResult):
        raise TypeError(f"Dense-ft training returned {type(result).__name__}.")
    return {"model_metadata": str(result.metadata_path)}


def _metric_records(result: object) -> list[dict[str, object]]:
    records = getattr(result, "metric_records", None)
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"Training result has invalid metric_records: {type(records).__name__}.")
    return list(records)


def _input_paths(config: TrainStageConfig) -> JsonObject:
    inputs: dict[str, JsonValue] = {
        "train_tasks": str(config.io.train_tasks),
        "train_labels": str(config.io.train_labels),
        "train_pairs": str(config.io.train_pairs),
        "dev_tasks": str(config.io.dev_tasks),
        "dev_labels": str(config.io.dev_labels),
    }
    if isinstance(config, RgcnTrainStageConfig):
        inputs["train_graphs"] = str(config.io.train_graphs)
        inputs["dev_graphs"] = str(config.io.dev_graphs)
        if config.io.train_labels is None:
            inputs.pop("train_labels")
    return inputs


def _output_paths(config: TrainStageConfig) -> JsonObject:
    best_checkpoint = (
        config.io.checkpoint_dir / "best.pt"
        if isinstance(config, RgcnTrainStageConfig)
        else config.io.model_dir
    )
    return {
        "best_checkpoint": str(best_checkpoint),
        "metrics": str(config.io.metrics),
        "run_summary": str(config.io.run_summary),
    }


def _effective_config(config: TrainStageConfig, result: object) -> JsonObject:
    effective: dict[str, JsonValue] = {
        "method": config.method.value,
        "training_config_path": str(config.io.config) if config.io.config is not None else None,
    }
    if isinstance(result, TrainableTrainingResult):
        effective["model_config"] = cast(JsonObject, result.model_config.to_json_dict())
        effective["training_config"] = cast(JsonObject, result.training_config.to_json_dict())
    else:
        effective["job"] = cast(JsonObject, _json_ready(config.job))
    return effective


def _result_counts(payload: TrainPayload, result: object) -> JsonObject:
    counts: dict[str, JsonValue] = {
        "train_tasks": len(payload.train_task_inputs),
        "train_pairs": len(payload.train_pairs),
        "dev_tasks": len(payload.dev_task_inputs),
    }
    if isinstance(result, TrainableTrainingResult):
        counts["epochs"] = result.training_config.epochs
        counts["global_step"] = result.global_step
    return counts


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
