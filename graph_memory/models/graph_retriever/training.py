from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from graph_memory.contracts.graphs import EvidenceGraph
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.service import evaluate_results
from graph_memory.models.graph_retriever.batching import (
    EvidenceTaskTensor,
    build_evidence_dataloader,
    materialize_full_ranking_tasks,
    materialize_training_tasks,
    move_training_batch,
)
from graph_memory.models.graph_retriever.config.records import (
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.dev_evaluation import predict_dev_from_batches
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.selection import (
    RgcnSelectionSettings,
    build_selection_metrics,
    is_selection_improvement,
    resolve_selection_metric,
)
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignalProvider


MetricRecord: TypeAlias = dict[str, object]
CheckpointCallback: TypeAlias = Callable[["RgcnTrainingResult"], None]


@dataclass(frozen=True)
class RgcnTrainingResult:
    """
    In-memory result of one trainable retriever training run.
    一次可训练检索器训练运行的内存结果。
    """

    model_config: RgcnModelConfig
    training_config: RgcnTrainingConfig
    metric_records: list[MetricRecord]
    best_model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object]
    scheduler_state_dict: dict[str, object]
    best_epoch: int
    global_step: int
    best_dev_metric: float


def train_graph_retriever(
    *,
    train_requests: list[TextRankingRequest],
    train_graphs: list[EvidenceGraph],
    train_pairs: list[TrainPairRecord],
    dev_requests: list[TextRankingRequest],
    dev_labels: list[EvidenceLabel],
    dev_graphs: list[EvidenceGraph],
    model_config: RgcnModelConfig,
    training_config: RgcnTrainingConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    selection_settings: RgcnSelectionSettings = RgcnSelectionSettings(),
    checkpoint_callback: CheckpointCallback | None = None,
    device: str | torch.device = "cpu",
) -> RgcnTrainingResult:
    """
    Train a frozen-encoder R-GCN binary node scorer.
    训练一个 frozen-encoder R-GCN 二分类节点 scorer。
    """

    _ = torch.manual_seed(training_config.random_seed)
    device = torch.device(device)
    model = build_model_from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    train_tasks = materialize_training_tasks(
        ranking_requests=train_requests,
        graphs=train_graphs,
        pairs=train_pairs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
    )
    if not train_tasks:
        raise ValueError("Training requires at least one supervised task tensor.")
    dev_tasks = materialize_full_ranking_tasks(
        ranking_requests=dev_requests,
        graphs=dev_graphs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        labels=dev_labels,
    )
    train_loader = build_evidence_dataloader(
        train_tasks,
        per_device_graph_batch_size=training_config.per_device_graph_batch_size,
        shuffle=True,
        random_seed=training_config.random_seed,
    )
    dev_loader = build_evidence_dataloader(
        dev_tasks,
        per_device_graph_batch_size=training_config.per_device_graph_batch_size,
        shuffle=False,
        random_seed=training_config.random_seed,
    )

    pos_weight = (
        _pos_weight(train_pairs, device) if training_config.pos_weight_enabled else None
    )
    metric_records: list[MetricRecord] = []
    best_metric = float("-inf") if selection_settings.higher_is_better else float("inf")
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    global_step = 0
    negative_count_by_type = _negative_count_by_type(train_pairs)
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)
    train_nodes_per_task = [len(task.graph_tensor.node_ids) for task in train_tasks]
    train_edges_per_task = [
        int(task.graph_tensor.edge_index.shape[1]) for task in train_tasks
    ]
    dev_nodes_per_task = [len(task.graph_tensor.node_ids) for task in dev_tasks]
    dev_edges_per_task = [
        int(task.graph_tensor.edge_index.shape[1]) for task in dev_tasks
    ]
    train_node_count = sum(train_nodes_per_task)
    train_edge_count = sum(train_edges_per_task)
    dev_node_count = sum(dev_nodes_per_task)
    dev_edge_count = sum(dev_edges_per_task)
    materialized_cpu_tensor_bytes = sum(
        _evidence_task_tensor_bytes(task) for task in [*train_tasks, *dev_tasks]
    )

    for epoch in tqdm(
        range(1, training_config.epochs + 1),
        desc="evidence-rgcn epochs",
        unit="epoch",
    ):
        model.train()
        _reset_peak_memory(device)
        train_started_at = perf_counter()
        train_loss_total = 0.0
        train_sample_count = 0
        optimizer_step_count = 0
        actual_tasks_per_optimizer_step: list[int] = []
        last_grad_norm = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            moved_batch = move_training_batch(batch, device)
            logits = model(moved_batch)
            loss_sum = F.binary_cross_entropy_with_logits(
                logits,
                moved_batch.labels,
                pos_weight=pos_weight,
                reduction="sum",
            )
            sample_count = int(moved_batch.labels.shape[0])
            if sample_count <= 0:
                raise ValueError("Evidence graph batch has no supervised samples.")
            (loss_sum / sample_count).backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), training_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            optimizer_step_count += 1
            task_count = moved_batch.graph_batch.task_count
            actual_tasks_per_optimizer_step.append(task_count)
            train_loss_total += float(loss_sum.detach().cpu())
            train_sample_count += sample_count
            last_grad_norm = float(
                grad_norm.detach().cpu()
                if isinstance(grad_norm, Tensor)
                else grad_norm
            )
        train_elapsed_seconds = perf_counter() - train_started_at
        train_peak_device_memory_bytes = _peak_memory(device)
        _reset_peak_memory(device)

        dev_predictions, dev_loss = predict_dev_from_batches(
            model=model,
            ranking_requests=dev_requests,
            labels=dev_labels,
            graphs=dev_graphs,
            model_config=model_config,
            batches=dev_loader,
            device=device,
        )
        dev_peak_device_memory_bytes = _peak_memory(device)
        dev_rows = evaluate_results(
            EvidenceEvaluationRequest(
                predictions=dev_predictions, labels=dev_labels, graphs=dev_graphs
            )
        )
        dev_row = dev_rows[0]
        selection_metrics = build_selection_metrics(
            dev_full_support_at_5=float(dev_row["Full Support@5"]),
            dev_full_support_at_10=float(dev_row["Full Support@10"]),
            dev_recall_at_5=float(dev_row["Recall@5"]),
            dev_mrr=float(dev_row["MRR"]),
            dev_loss=dev_loss,
        )
        dev_metric = resolve_selection_metric(selection_metrics, selection_settings)
        if is_selection_improvement(
            current=dev_metric,
            best=best_metric,
            settings=selection_settings,
        ):
            best_metric = dev_metric
            best_epoch = epoch
            best_state = _cpu_state_dict(model)

        metric_records.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_optimizer_step_count": optimizer_step_count,
                "per_device_graph_batch_size": (
                    training_config.per_device_graph_batch_size
                ),
                "actual_tasks_per_optimizer_step": actual_tasks_per_optimizer_step,
                "train_supervised_sample_count": train_sample_count,
                "train_task_count": len(train_tasks),
                "train_node_count": train_node_count,
                "train_edge_count": train_edge_count,
                "dev_task_count": len(dev_tasks),
                "dev_node_count": dev_node_count,
                "dev_edge_count": dev_edge_count,
                "materialized_cpu_tensor_bytes": materialized_cpu_tensor_bytes,
                "train_elapsed_seconds": train_elapsed_seconds,
                "train_tasks_per_second": (
                    len(train_tasks) / train_elapsed_seconds
                    if train_elapsed_seconds > 0.0
                    else 0.0
                ),
                "train_peak_device_memory_bytes": train_peak_device_memory_bytes,
                "dev_peak_device_memory_bytes": dev_peak_device_memory_bytes,
                **_count_stats("train_nodes_per_task", train_nodes_per_task),
                **_count_stats("train_edges_per_task", train_edges_per_task),
                **_count_stats("dev_nodes_per_task", dev_nodes_per_task),
                **_count_stats("dev_edges_per_task", dev_edges_per_task),
                "train_loss": train_loss_total / train_sample_count
                if train_sample_count
                else 0.0,
                "dev_loss": dev_loss,
                "dev_recall_at_5": float(dev_row["Recall@5"]),
                "dev_full_support_at_5": float(dev_row["Full Support@5"]),
                "dev_full_support_at_10": float(dev_row["Full Support@10"]),
                "dev_mrr": float(dev_row["MRR"]),
                "selection_metric": selection_settings.best_metric,
                "selection_metric_value": dev_metric,
                "best_dev_metric": best_metric,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": last_grad_norm,
                "positive_count": positive_count,
                "negative_count_by_type": negative_count_by_type,
            }
        )

    result = RgcnTrainingResult(
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


def _count_stats(prefix: str, values: list[int]) -> dict[str, float | int]:
    if not values:
        return {f"{prefix}_min": 0, f"{prefix}_max": 0, f"{prefix}_mean": 0.0}
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_mean": sum(values) / len(values),
    }


def _evidence_task_tensor_bytes(task: EvidenceTaskTensor) -> int:
    tensors = (
        task.graph_tensor.node_embeddings,
        task.graph_tensor.node_features,
        task.graph_tensor.edge_index,
        task.graph_tensor.relation_ids,
        task.graph_tensor.edge_weights,
        task.sample_node_indices,
        task.sample_node_features,
        task.labels,
    )
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _pos_weight(train_pairs: list[TrainPairRecord], device: torch.device) -> Tensor:
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)
    negative_count = sum(1 for pair in train_pairs if pair["label"] == 0)
    if positive_count == 0:
        raise ValueError("pos_weight requires at least one positive sample.")
    return torch.tensor(
        [negative_count / positive_count], dtype=torch.float32, device=device
    )


def _negative_count_by_type(train_pairs: list[TrainPairRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter(
        pair["sample_type"] for pair in train_pairs if pair["label"] == 0
    )
    return dict(sorted(counter.items()))


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in deepcopy(model.state_dict()).items()
    }
