from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.config import CONFIG_LOADER
from graph_memory.contracts.common import JsonObject
from graph_memory.io import read_json, write_jsonl
from graph_memory.models.graph_retriever.checkpoint import save_trainable_checkpoint
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.text_embeddings import DenseTextEmbeddingProvider
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.registry import Registry
from graph_memory.registry.stage_configs import TrainStageConfig
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider, SeedSignalProvider
from graph_memory.stages.train import run_train_stage

LOGGER = logging.getLogger("train_graph_retriever")


@dataclass(frozen=True)
class TrainGraphRetrieverArgs:
    """
    Parsed CLI arguments for trainable graph retriever training.
    可训练图检索器训练脚本的 CLI 参数。

    Fields / 字段:
    - train_tasks: Path to train task input artifact.
      train_tasks：train task input artifact 路径。
    - train_labels: Path to train label artifact.
      train_labels：train label artifact 路径。
    - train_graphs: Path to train graph artifact.
      train_graphs：train graph artifact 路径。
    - train_pairs: Path to train pair artifact.
      train_pairs：train pair artifact 路径。
    - dev_tasks: Path to dev task input artifact.
      dev_tasks：dev task input artifact 路径。
    - dev_labels: Path to dev label artifact.
      dev_labels：dev label artifact 路径。
    - dev_graphs: Path to dev graph artifact.
      dev_graphs：dev graph artifact 路径。
    - output_dir: Directory for checkpoints, metrics, and run summary.
      output_dir：checkpoint、metrics 和 run summary 输出目录。
    - encoder_model: Frozen text encoder model name or local path.
      encoder_model：冻结文本 encoder 模型名或本地路径。
    - query_prefix: Query encoding prefix.
      query_prefix：query 编码前缀。
    - passage_prefix: Passage encoding prefix.
      passage_prefix：passage 编码前缀。
    - hidden_dim: Hidden dimension.
      hidden_dim：隐藏维度。
    - num_layers: Number of R-GCN layers.
      num_layers：R-GCN 层数。
    - dropout: Dropout probability.
      dropout：dropout 概率。
    - ablation: Canonical ablation name.
      ablation：规范化 ablation 名称。
    - epochs: Number of epochs.
      epochs：epoch 数量。
    - batch_size: Task graphs per batch.
      batch_size：每个 batch 的 task graph 数量。
    - learning_rate: AdamW learning rate.
      learning_rate：AdamW 学习率。
    - max_grad_norm: Gradient clipping max norm.
      max_grad_norm：梯度裁剪最大 norm。
    - random_seed: Run random seed.
      random_seed：运行随机种子。
    - pos_weight: Whether to enable BCE pos_weight.
      pos_weight：是否启用 BCE pos_weight。
    - device: Torch device.
      device：torch device。
    - config: Optional resolved trainable training config path.
      config：可选的已解析可训练 training config 路径。
    """

    train_tasks: str
    train_labels: str
    train_graphs: str
    train_pairs: str
    dev_tasks: str
    dev_labels: str
    dev_graphs: str
    output_dir: str
    encoder_model: str
    query_prefix: str
    passage_prefix: str
    hidden_dim: int
    num_layers: int
    dropout: float
    ablation: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    random_seed: int
    pos_weight: bool
    device: str
    config: str | None


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
        "train_labels": str(config.io.train_labels),
        "train_graphs": str(config.io.train_graphs),
        "train_pairs": str(config.io.train_pairs),
        "dev_tasks": str(config.io.dev_tasks),
        "dev_labels": str(config.io.dev_labels),
        "dev_graphs": str(config.io.dev_graphs),
    }
    outputs = {
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "metrics": str(metrics_path),
        "run_summary": str(run_summary_path),
    }

    try:
        embedding_provider = _text_embedding_provider_from_config(config, text_embedding_provider)
        seed_provider = _seed_signal_provider_from_config(config, seed_signal_provider, embedding_provider)
        train_task_inputs = read_json(config.io.train_tasks)
        train_graphs = read_json(config.io.train_graphs)
        train_pairs = read_json(config.io.train_pairs)
        dev_task_inputs = read_json(config.io.dev_tasks)
        dev_labels = read_json(config.io.dev_labels)
        dev_graphs = read_json(config.io.dev_graphs)
        result = run_train_stage(
            config,
            train_task_inputs=train_task_inputs,
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


def build_parser() -> argparse.ArgumentParser:
    return Registry.configs.TRAIN.parser_factory()


def parse_args(argv: Sequence[str] | None = None) -> TrainGraphRetrieverArgs:
    namespace = build_parser().parse_args(argv)
    return TrainGraphRetrieverArgs(
        train_tasks=namespace.train_tasks,
        train_labels=namespace.train_labels,
        train_graphs=namespace.train_graphs,
        train_pairs=namespace.train_pairs,
        dev_tasks=namespace.dev_tasks,
        dev_labels=namespace.dev_labels,
        dev_graphs=namespace.dev_graphs,
        output_dir=namespace.output_dir,
        encoder_model=namespace.encoder_model,
        query_prefix=namespace.query_prefix,
        passage_prefix=namespace.passage_prefix,
        hidden_dim=namespace.hidden_dim,
        num_layers=namespace.num_layers,
        dropout=namespace.dropout,
        ablation=namespace.ablation,
        epochs=namespace.epochs,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        max_grad_norm=namespace.max_grad_norm,
        random_seed=namespace.random_seed,
        pos_weight=namespace.pos_weight,
        device=namespace.device,
        config=namespace.config,
    )


def _text_embedding_provider_from_config(
    config: TrainStageConfig,
    provider: TextEmbeddingProvider | None,
) -> TextEmbeddingProvider:
    if provider is not None:
        return provider
    encoder = config.job.encoder
    return DenseTextEmbeddingProvider(
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
