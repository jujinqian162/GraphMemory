from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import cast
from typing_extensions import assert_never

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonObject, JsonValue
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.datasets.selection import evidence_labels_for_dataset, text_ranking_requests_for_dataset
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.dense_finetune.training import DenseFinetuneTrainingResult
from graph_memory.models.graph_retriever.checkpoint import save_rgcn_checkpoint
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.training import RgcnTrainingResult
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import (
    DenseFinetuneTrainStageConfig,
    RgcnTrainStageConfig,
    TrainStageConfig,
)
from graph_memory.stages.train_payloads import (
    DenseFinetuneTrainPayload,
    RgcnTrainPayload,
    TrainDependencies,
    TrainPayload,
)
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.stages.train import TrainingResult, run_train_stage

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
            effective_config={"dataset": config.dataset, "method": config.method.value},
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
        train_records = cast(list[object], read_json(config.io.train_tasks))
        train_labels = (
            None
            if config.io.train_labels is None
            else _evidence_labels(config, cast(list[object], read_json(config.io.train_labels)))
        )
        dev_records = cast(list[object], read_json(config.io.dev_tasks))
        return RgcnTrainPayload(
            train_requests=_text_requests(config, train_records),
            train_labels=train_labels,
            train_graphs=cast(list[MemoryGraph], read_json(config.io.train_graphs)),
            train_pairs=cast(list[TrainPairRecord], read_json(config.io.train_pairs)),
            dev_requests=_text_requests(config, dev_records),
            dev_labels=_evidence_labels(config, cast(list[object], read_json(config.io.dev_labels))),
            dev_graphs=cast(list[MemoryGraph], read_json(config.io.dev_graphs)),
            dependencies=dependencies,
        )
    if isinstance(config, DenseFinetuneTrainStageConfig):
        if dependencies is not None:
            raise ValueError("Provider overrides are only valid for R-GCN training.")
        train_records = cast(list[object], read_json(config.io.train_tasks))
        dev_records = cast(list[object], read_json(config.io.dev_tasks))
        return DenseFinetuneTrainPayload(
            train_requests=_text_requests(config, train_records),
            train_labels=_evidence_labels(config, cast(list[object], read_json(config.io.train_labels))),
            train_pairs=cast(list[TrainPairRecord], read_json(config.io.train_pairs)),
            dev_requests=_text_requests(config, dev_records),
            dev_labels=_evidence_labels(config, cast(list[object], read_json(config.io.dev_labels))),
            output_dir=config.io.output_dir,
            model_dir=config.io.model_dir,
        )
    assert_never(config)


def _text_requests(config: TrainStageConfig, records: Sequence[object]) -> list[TextRankingRequest]:
    return text_ranking_requests_for_dataset(config.dataset, records)


def _evidence_labels(config: TrainStageConfig, labels: Sequence[object]) -> list[EvidenceLabel]:
    return evidence_labels_for_dataset(config.dataset, labels)


def _write_method_artifacts(config: TrainStageConfig, result: TrainingResult) -> JsonObject:
    if isinstance(config, RgcnTrainStageConfig):
        if not isinstance(result, RgcnTrainingResult):
            raise TypeError(f"R-GCN training returned {type(result).__name__}.")
        best_model = build_model_from_config(result.model_config)
        best_model.load_state_dict(result.best_model_state_dict)
        epoch_checkpoint = config.io.checkpoint_dir / f"checkpoint_epoch_{result.best_epoch}.pt"
        for path in (epoch_checkpoint, config.io.checkpoint_dir / "best.pt"):
            save_rgcn_checkpoint(
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
    if isinstance(config, DenseFinetuneTrainStageConfig):
        if not isinstance(result, DenseFinetuneTrainingResult):
            raise TypeError(f"Dense-FT training returned {type(result).__name__}.")
        return {"model_metadata": str(result.metadata_path)}
    assert_never(config)


def _metric_records(result: TrainingResult) -> list[dict[str, object]]:
    if isinstance(result, RgcnTrainingResult):
        return list(result.metric_records)
    if isinstance(result, DenseFinetuneTrainingResult):
        return list(result.metric_records)
    assert_never(result)


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
    elif isinstance(config, DenseFinetuneTrainStageConfig):
        pass
    else:
        assert_never(config)
    return inputs


def _output_paths(config: TrainStageConfig) -> JsonObject:
    if isinstance(config, RgcnTrainStageConfig):
        best_checkpoint = config.io.checkpoint_dir / "best.pt"
    elif isinstance(config, DenseFinetuneTrainStageConfig):
        best_checkpoint = config.io.model_dir
    else:
        assert_never(config)
    return {
        "best_checkpoint": str(best_checkpoint),
        "metrics": str(config.io.metrics),
        "run_summary": str(config.io.run_summary),
    }


def _effective_config(config: TrainStageConfig, result: TrainingResult) -> JsonObject:
    effective: dict[str, JsonValue] = {
        "dataset": config.dataset,
        "method": config.method.value,
    }
    if isinstance(config, RgcnTrainStageConfig):
        if not isinstance(result, RgcnTrainingResult):
            raise TypeError(f"R-GCN training returned {type(result).__name__}.")
        effective["model_config"] = cast(JsonObject, result.model_config.to_json_dict())
        effective["training_config"] = cast(JsonObject, result.training_config.to_json_dict())
    elif isinstance(config, DenseFinetuneTrainStageConfig):
        if not isinstance(result, DenseFinetuneTrainingResult):
            raise TypeError(f"Dense-FT training returned {type(result).__name__}.")
        effective["job"] = cast(JsonObject, _json_ready(config.job))
    else:
        assert_never(config)
    return effective


def _result_counts(payload: TrainPayload, result: TrainingResult) -> JsonObject:
    counts: dict[str, JsonValue] = {
        "train_tasks": len(payload.train_requests),
        "train_pairs": len(payload.train_pairs),
        "dev_tasks": len(payload.dev_requests),
    }
    if isinstance(result, RgcnTrainingResult):
        counts["epochs"] = result.training_config.epochs
        counts["global_step"] = result.global_step
    elif isinstance(result, DenseFinetuneTrainingResult):
        pass
    else:
        assert_never(result)
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
