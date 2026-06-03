from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.evaluation.service import evaluate_results
from graph_memory.models.graph_retriever.batching import build_training_batches, move_training_batch
from graph_memory.models.graph_retriever.config.records import TrainableModelConfig, TrainableTrainingConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.dev_evaluation import best_metric as select_best_metric
from graph_memory.models.graph_retriever.dev_evaluation import predict_dev
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.validation import (
    validate_graphs,
    validate_memory_task_inputs,
    validate_memory_task_labels,
    validate_train_pairs,
    validate_trainable_model_config,
    validate_trainable_training_config,
)


MetricRecord: TypeAlias = dict[str, object]
CheckpointCallback: TypeAlias = Callable[["TrainableTrainingResult"], None]


@dataclass(frozen=True)
class TrainableTrainingResult:
    """
    In-memory result of one trainable retriever training run.
    一次可训练检索器训练运行的内存结果。

    Fields / 字段:
    - model_config: Effective model config used by the run.
      model_config：本次运行使用的实际模型配置。
    - training_config: Effective training config used by the run.
      training_config：本次运行使用的实际训练配置。
    - metric_records: Epoch-level metric records ready for JSONL output.
      metric_records：可写入 JSONL 的 epoch 级 metric 记录。
    - best_model_state_dict: CPU state dict for the best dev checkpoint.
      best_model_state_dict：最佳 dev checkpoint 对应的 CPU state dict。
    - optimizer_state_dict: Optimizer state dict after training.
      optimizer_state_dict：训练结束后的 optimizer state dict。
    - scheduler_state_dict: Scheduler state dict after training.
      scheduler_state_dict：训练结束后的 scheduler state dict。
    - best_epoch: Epoch number with the best dev metric.
      best_epoch：dev metric 最好的 epoch 编号。
    - global_step: Total optimizer steps.
      global_step：optimizer step 总数。
    - best_dev_metric: Best checkpoint selection metric.
      best_dev_metric：最佳 checkpoint 选择指标。
    """

    model_config: TrainableModelConfig
    training_config: TrainableTrainingConfig
    metric_records: list[MetricRecord]
    best_model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object]
    best_epoch: int
    global_step: int
    best_dev_metric: float


def train_graph_retriever(
    *,
    train_task_inputs: list[MemoryTaskInput],
    train_graphs: list[MemoryGraph],
    train_pairs: list[TrainPairRecord],
    dev_task_inputs: list[MemoryTaskInput],
    dev_labels: list[MemoryTaskLabels],
    dev_graphs: list[MemoryGraph],
    model_config: TrainableModelConfig,
    training_config: TrainableTrainingConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    train_labels: list[MemoryTaskLabels] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    device: str | torch.device = "cpu",
) -> TrainableTrainingResult:
    """
    Train a frozen-encoder R-GCN binary node scorer.
    训练一个 frozen-encoder R-GCN 二分类节点 scorer。
    """

    validate_trainable_model_config(model_config)
    validate_trainable_training_config(training_config)
    validate_memory_task_inputs(train_task_inputs)
    validate_memory_task_inputs(dev_task_inputs)
    train_inputs_by_task_id = {task_input["task_id"]: task_input for task_input in train_task_inputs}
    dev_inputs_by_task_id = {task_input["task_id"]: task_input for task_input in dev_task_inputs}
    validate_graphs(train_graphs, train_inputs_by_task_id)
    validate_graphs(dev_graphs, dev_inputs_by_task_id)
    validate_memory_task_labels(dev_labels, dev_inputs_by_task_id)
    if train_labels is not None:
        train_labels_by_task_id = {label["task_id"]: label for label in train_labels}
        validate_memory_task_labels(train_labels, train_inputs_by_task_id)
        validate_train_pairs(
            train_pairs,
            train_inputs_by_task_id,
            train_labels_by_task_id,
            {graph["task_id"]: graph for graph in train_graphs},
        )

    _ = torch.manual_seed(training_config.random_seed)
    device = torch.device(device)
    model = build_model_from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    train_batches = build_training_batches(
        task_inputs=train_task_inputs,
        graphs=train_graphs,
        pairs=train_pairs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=training_config.batch_size,
    )
    if not train_batches:
        raise ValueError("Training requires at least one non-empty training batch.")

    pos_weight = _pos_weight(train_pairs, device) if training_config.pos_weight_enabled else None
    metric_records: list[MetricRecord] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    global_step = 0
    negative_count_by_type = _negative_count_by_type(train_pairs)
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_sample_count = 0
        last_grad_norm = 0.0
        for batch in train_batches:
            moved_batch = move_training_batch(batch, device)
            logits = model(moved_batch)
            loss = F.binary_cross_entropy_with_logits(logits, moved_batch.labels, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            sample_count = int(moved_batch.labels.shape[0])
            train_loss_total += float(loss.detach().cpu()) * sample_count
            train_sample_count += sample_count
            last_grad_norm = float(grad_norm.detach().cpu() if isinstance(grad_norm, Tensor) else grad_norm)

        dev_predictions, dev_loss = predict_dev(
            model=model,
            task_inputs=dev_task_inputs,
            labels=dev_labels,
            graphs=dev_graphs,
            model_config=model_config,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            batch_size=training_config.batch_size,
            device=device,
        )
        dev_rows = evaluate_results(dev_predictions, dev_labels, dev_graphs)
        dev_row = dev_rows[0]
        dev_metric = select_best_metric(dev_row)
        if dev_metric > best_metric:
            best_metric = dev_metric
            best_epoch = epoch
            best_state = _cpu_state_dict(model)

        metric_records.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss_total / train_sample_count if train_sample_count else 0.0,
                "dev_loss": dev_loss,
                "dev_recall_at_5": float(dev_row["Recall@5"]),
                "dev_full_support_at_5": float(dev_row["Full Support@5"]),
                "dev_full_support_at_10": float(dev_row["Full Support@10"]),
                "dev_mrr": float(dev_row["MRR"]),
                "best_dev_metric": best_metric,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": last_grad_norm,
                "positive_count": positive_count,
                "negative_count_by_type": negative_count_by_type,
            }
        )

    result = TrainableTrainingResult(
        model_config=model_config,
        training_config=training_config,
        metric_records=metric_records,
        best_model_state_dict=best_state,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        best_epoch=best_epoch,
        global_step=global_step,
        best_dev_metric=best_metric,
    )
    if checkpoint_callback is not None:
        checkpoint_callback(result)
    return result


def _pos_weight(train_pairs: list[TrainPairRecord], device: torch.device) -> Tensor:
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)
    negative_count = sum(1 for pair in train_pairs if pair["label"] == 0)
    if positive_count == 0:
        raise ValueError("pos_weight requires at least one positive sample.")
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)


def _negative_count_by_type(train_pairs: list[TrainPairRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter(pair["sample_type"] for pair in train_pairs if pair["label"] == 0)
    return dict(sorted(counter.items()))


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in deepcopy(model.state_dict()).items()}
