from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.service import evaluate_results
from graph_memory.models.graph_retriever.batching import (
    build_full_ranking_batches,
    build_training_batches,
    move_training_batch,
)
from graph_memory.models.graph_retriever.config.records import RgcnModelConfig, RgcnTrainingConfig
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.dev_evaluation import best_metric as select_best_metric
from graph_memory.models.graph_retriever.dev_evaluation import predict_dev_from_batches
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignalProvider
from graph_memory.validation import (
    validate_graphs,
    validate_rgcn_model_config,
    validate_rgcn_training_config,
    validate_train_pairs,
)


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
    train_graphs: list[MemoryGraph],
    train_pairs: list[TrainPairRecord],
    dev_requests: list[TextRankingRequest],
    dev_labels: list[EvidenceLabel],
    dev_graphs: list[MemoryGraph],
    model_config: RgcnModelConfig,
    training_config: RgcnTrainingConfig,
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    train_labels: list[EvidenceLabel] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    device: str | torch.device = "cpu",
) -> RgcnTrainingResult:
    """
    Train a frozen-encoder R-GCN binary node scorer.
    训练一个 frozen-encoder R-GCN 二分类节点 scorer。
    """

    validate_rgcn_model_config(model_config)
    validate_rgcn_training_config(training_config)
    validate_graphs(train_graphs, train_requests)
    validate_graphs(dev_graphs, dev_requests)
    if train_labels is not None:
        validate_train_pairs(
            train_pairs,
            train_requests,
            train_labels,
            {graph["task_id"]: graph for graph in train_graphs},
        )

    _ = torch.manual_seed(training_config.random_seed)
    device = torch.device(device)
    model = build_model_from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    train_batches = build_training_batches(
        ranking_requests=train_requests,
        graphs=train_graphs,
        pairs=train_pairs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=training_config.batch_size,
    )
    if not train_batches:
        raise ValueError("Training requires at least one non-empty training batch.")
    dev_batches = build_full_ranking_batches(
        ranking_requests=dev_requests,
        graphs=dev_graphs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=training_config.batch_size,
        labels=dev_labels,
    )

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

        dev_predictions, dev_loss = predict_dev_from_batches(
            model=model,
            ranking_requests=dev_requests,
            labels=dev_labels,
            graphs=dev_graphs,
            model_config=model_config,
            batches=dev_batches,
            device=device,
        )
        dev_rows = evaluate_results(EvidenceEvaluationRequest(predictions=dev_predictions, labels=dev_labels, graphs=dev_graphs))
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
