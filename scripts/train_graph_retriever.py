from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonObject
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.graph_retriever.checkpoint import save_trainable_checkpoint
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.text_embeddings import DenseGraphFeatureProvider
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import TrainStageConfig
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider, SeedSignalProvider
from graph_memory.stages.train import run_train_stage

LOGGER = logging.getLogger("train_graph_retriever")


def main(
    argv: Sequence[str] | None = None,
    *,
    text_embedding_provider: TextEmbeddingProvider | None = None,
    seed_signal_provider: SeedSignalProvider | None = None,
) -> int:
    config = CONFIG_LOADER.load(Registry.configs.TRAIN, argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_dir = config.io.output_dir
    checkpoint_dir = config.io.checkpoint_dir
    metrics_path = config.io.metrics
    run_summary_path = config.io.run_summary
    inputs = {
        "train_tasks": str(config.io.train_tasks),
        "train_graphs": str(config.io.train_graphs),
        "train_pairs": str(config.io.train_pairs),
        "dev_tasks": str(config.io.dev_tasks),
        "dev_labels": str(config.io.dev_labels),
        "dev_graphs": str(config.io.dev_graphs),
    }
    if config.io.train_labels is not None:
        inputs["train_labels"] = str(config.io.train_labels)
    outputs = {
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "metrics": str(metrics_path),
        "run_summary": str(run_summary_path),
    }

    try:
        embedding_provider = _text_embedding_provider_from_config(config, text_embedding_provider)
        seed_provider = _seed_signal_provider_from_config(config, seed_signal_provider, embedding_provider)
        train_task_inputs = read_json(config.io.train_tasks)
        train_labels = read_json(config.io.train_labels) if config.io.train_labels is not None else None
        train_graphs = read_json(config.io.train_graphs)
        train_pairs = read_json(config.io.train_pairs)
        dev_task_inputs = read_json(config.io.dev_tasks)
        dev_labels = read_json(config.io.dev_labels)
        dev_graphs = read_json(config.io.dev_graphs)
        result = run_train_stage(
            config,
            train_task_inputs=train_task_inputs,
            train_labels=train_labels,
            train_graphs=train_graphs,
            train_pairs=train_pairs,
            dev_task_inputs=dev_task_inputs,
            dev_labels=dev_labels,
            dev_graphs=dev_graphs,
            text_embedding_provider=embedding_provider,
            seed_signal_provider=seed_provider,
        )
        training_result = result.result
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(metrics_path, training_result.metric_records)

        best_model = build_model_from_config(training_result.model_config)
        best_model.load_state_dict(training_result.best_model_state_dict)
        epoch_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{training_result.best_epoch}.pt"
        save_trainable_checkpoint(
            epoch_checkpoint_path,
            method_name=training_result.model_config.method_name,
            model=best_model,
            optimizer_state_dict=training_result.optimizer_state_dict,
            scheduler_state_dict=training_result.scheduler_state_dict,
            epoch=training_result.best_epoch,
            global_step=training_result.global_step,
            best_dev_metric=training_result.best_dev_metric,
            model_config=training_result.model_config,
            training_config=training_result.training_config,
        )
        save_trainable_checkpoint(
            checkpoint_dir / "best.pt",
            method_name=training_result.model_config.method_name,
            model=best_model,
            optimizer_state_dict=training_result.optimizer_state_dict,
            scheduler_state_dict=training_result.scheduler_state_dict,
            epoch=training_result.best_epoch,
            global_step=training_result.global_step,
            best_dev_metric=training_result.best_dev_metric,
            model_config=training_result.model_config,
            training_config=training_result.training_config,
        )

        summary = build_run_summary(
            script="train_graph_retriever.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=_effective_config(config, training_result),
            inputs=inputs,
            outputs={**outputs, "epoch_checkpoint": str(epoch_checkpoint_path)},
            counts={
                "train_tasks": len(train_task_inputs),
                "train_pairs": len(train_pairs),
                "dev_tasks": len(dev_task_inputs),
                "epochs": training_result.training_config.epochs,
                "global_step": training_result.global_step,
            },
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
        )
        write_run_summary(run_summary_path, summary)
        LOGGER.info("wrote best checkpoint: %s", checkpoint_dir / "best.pt")
        LOGGER.info("wrote metrics: %s", metrics_path)
        LOGGER.info("wrote run summary: %s", run_summary_path)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        summary = build_run_summary(
            script="train_graph_retriever.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="failed",
            effective_config={},
            inputs=inputs,
            outputs=outputs,
            counts={},
            timings={"total_seconds": time.perf_counter() - start_time},
            environment=collect_environment(),
            notes=[],
            error=str(error),
        )
        write_run_summary(run_summary_path, summary)
        raise


def _text_embedding_provider_from_config(
    config: TrainStageConfig,
    provider: TextEmbeddingProvider | None,
) -> TextEmbeddingProvider:
    if provider is not None:
        return provider
    encoder = config.job.encoder
    return DenseGraphFeatureProvider(
        model_name=encoder.model_name,
        query_prefix=encoder.query_prefix,
        passage_prefix=encoder.passage_prefix,
        batch_size=encoder.batch_size,
    )


def _seed_signal_provider_from_config(
    config: TrainStageConfig,
    provider: SeedSignalProvider | None,
    embedding_provider: TextEmbeddingProvider,
) -> SeedSignalProvider:
    if provider is not None:
        return provider
    if isinstance(embedding_provider, DenseGraphFeatureProvider):
        return embedding_provider
    encoder = config.job.encoder
    return RetrieverSeedSignalProvider(
        DenseTaskRetriever(
            config=DenseConfig(
                model_name=encoder.model_name,
                query_prefix=encoder.query_prefix,
                passage_prefix=encoder.passage_prefix,
                batch_size=encoder.batch_size,
            ),
            encoder=getattr(embedding_provider, "encoder", None),
        )
    )


def _effective_config(config: TrainStageConfig, result: object) -> JsonObject:
    model_config = getattr(result, "model_config")
    training_config = getattr(result, "training_config")
    return cast(
        JsonObject,
        {
            "model_config": model_config.to_json_dict(),
            "training_config": training_config.to_json_dict(),
            "training_config_path": str(config.io.config) if config.io.config is not None else None,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
