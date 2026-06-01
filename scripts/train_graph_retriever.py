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

from graph_memory.indexes.dense import DenseTaskRetriever
from graph_memory.io import read_json, write_jsonl
from graph_memory.learned.checkpoint import save_trainable_checkpoint
from graph_memory.learned.features import (
    DenseTextEmbeddingProvider,
    RetrieverSeedSignalProvider,
    SeedSignalProvider,
    TextEmbeddingProvider,
)
from graph_memory.learned.training import build_model_from_config, default_model_config, train_graph_retriever
from graph_memory.observability import build_run_summary, collect_environment, now_iso, write_run_summary
from graph_memory.training_config import (
    EncoderConfig,
    JsonConfig,
    ModelConfigValues,
    device_from_training_config,
    encoder_config_from_training_config,
    load_trainable_training_config,
    model_config_values_from_training_config,
    trainable_training_config_from_training_config,
)
from graph_memory.types import JsonObject, TrainableTrainingConfig

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
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    started_at = now_iso()
    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "train_metrics.jsonl"
    run_summary_path = output_dir / "train_run_summary.json"
    inputs = {
        "train_tasks": args.train_tasks,
        "train_labels": args.train_labels,
        "train_graphs": args.train_graphs,
        "train_pairs": args.train_pairs,
        "dev_tasks": args.dev_tasks,
        "dev_labels": args.dev_labels,
        "dev_graphs": args.dev_graphs,
    }
    outputs = {
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "metrics": str(metrics_path),
        "run_summary": str(run_summary_path),
    }

    try:
        file_config = load_trainable_training_config(args.config) if args.config is not None else None
        encoder_config = _encoder_config_from_args(args, file_config)
        model_values = _model_values_from_args(args, file_config)
        training_config = _training_config_from_args(args, file_config)
        device = device_from_training_config(file_config) if file_config is not None else args.device

        embedding_provider = text_embedding_provider
        if embedding_provider is None:
            embedding_provider = DenseTextEmbeddingProvider(
                model_name=encoder_config["model"],
                query_prefix=encoder_config["query_prefix"],
                passage_prefix=encoder_config["passage_prefix"],
            )
        seed_provider = seed_signal_provider
        if seed_provider is None:
            seed_provider = RetrieverSeedSignalProvider(
                DenseTaskRetriever(
                    model_name=encoder_config["model"],
                    query_prefix=encoder_config["query_prefix"],
                    passage_prefix=encoder_config["passage_prefix"],
                    encoder=getattr(embedding_provider, "encoder", None),
                )
            )
        model_config = default_model_config(
            encoder_model=encoder_config["model"],
            encoder_dim=embedding_provider.embedding_dim,
            query_prefix=encoder_config["query_prefix"],
            passage_prefix=encoder_config["passage_prefix"],
            hidden_dim=model_values["hidden_dim"],
            num_layers=model_values["num_layers"],
            dropout=model_values["dropout"],
            ablation_name=model_values["ablation"],
        )
        result = train_graph_retriever(
            train_task_inputs=read_json(args.train_tasks),
            train_labels=read_json(args.train_labels),
            train_graphs=read_json(args.train_graphs),
            train_pairs=read_json(args.train_pairs),
            dev_task_inputs=read_json(args.dev_tasks),
            dev_labels=read_json(args.dev_labels),
            dev_graphs=read_json(args.dev_graphs),
            model_config=model_config,
            training_config=training_config,
            text_embedding_provider=embedding_provider,
            seed_signal_provider=seed_provider,
            device=device,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(metrics_path, result.metric_records)

        best_model = build_model_from_config(result.model_config)
        best_model.load_state_dict(result.best_model_state_dict)
        epoch_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{result.best_epoch}.pt"
        save_trainable_checkpoint(
            epoch_checkpoint_path,
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
        save_trainable_checkpoint(
            checkpoint_dir / "best.pt",
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

        effective_config = cast(
            JsonObject,
            {
                "model_config": model_config.to_json_dict(),
                "training_config": training_config.to_json_dict(),
                "training_config_path": args.config,
            },
        )
        summary = build_run_summary(
            script="train_graph_retriever.py",
            started_at=started_at,
            finished_at=now_iso(),
            status="success",
            effective_config=effective_config,
            inputs=inputs,
            outputs={**outputs, "epoch_checkpoint": str(epoch_checkpoint_path)},
            counts={
                "train_tasks": len(read_json(args.train_tasks)),
                "train_pairs": len(read_json(args.train_pairs)),
                "dev_tasks": len(read_json(args.dev_tasks)),
                "epochs": training_config.epochs,
                "global_step": result.global_step,
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
    parser = argparse.ArgumentParser(description="Train the Phase 2 R-GCN graph retriever.")
    parser.add_argument("--train_tasks", required=True)
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--train_graphs", required=True)
    parser.add_argument("--train_pairs", required=True)
    parser.add_argument("--dev_tasks", required=True)
    parser.add_argument("--dev_labels", required=True)
    parser.add_argument("--dev_graphs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder_model", default="intfloat/e5-base-v2")
    parser.add_argument("--query_prefix", default="query: ")
    parser.add_argument("--passage_prefix", default="passage: ")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ablation",
        default="full_rgcn",
        choices=["full_rgcn", "wo_graph", "wo_edge_type", "wo_bridge", "wo_edge_weight", "wo_seed_score"],
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=13)
    parser.add_argument("--pos_weight", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None, help="Path to resolved trainable training config JSON.")
    return parser


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


def _encoder_config_from_args(args: TrainGraphRetrieverArgs, config: JsonConfig | None) -> EncoderConfig:
    if config is not None:
        return encoder_config_from_training_config(config)
    return {
        "model": args.encoder_model,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
    }


def _model_values_from_args(args: TrainGraphRetrieverArgs, config: JsonConfig | None) -> ModelConfigValues:
    if config is not None:
        return model_config_values_from_training_config(config)
    return {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "ablation": args.ablation,
    }


def _training_config_from_args(
    args: TrainGraphRetrieverArgs,
    config: JsonConfig | None,
) -> TrainableTrainingConfig:
    if config is not None:
        return trainable_training_config_from_training_config(config)
    return TrainableTrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        random_seed=args.random_seed,
        pos_weight_enabled=args.pos_weight,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
