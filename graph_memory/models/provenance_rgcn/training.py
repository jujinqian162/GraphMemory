from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from graph_memory.embeddings import SentenceEncoder
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.provenance_rgcn.config import (
    ProvenanceRgcnModelConfig,
    ProvenanceRgcnTrainingConfig,
)
from graph_memory.models.provenance_rgcn.contracts import (
    ProvenanceGraphTensor,
    ProvenanceModelOutput,
)
from graph_memory.models.provenance_rgcn.model import ExecutionProvenanceRGCN
from graph_memory.models.provenance_rgcn.inference import (
    ExecutionProvenanceRgcnRetriever,
)
from graph_memory.models.provenance_rgcn.tensorization import (
    move_provenance_tensor,
    tensorize_provenance_request,
)
from graph_memory.retrieval.requests import ExecutionProvenanceRankingRequest
from graph_memory.retrieval.contracts import ExecutionProvenanceTrace


@dataclass(frozen=True)
class ProvenanceLoss:
    total: Tensor
    candidate: Tensor
    edge: Tensor
    comparison_count: int


@dataclass(frozen=True)
class ProvenanceTrainingResult:
    model: ExecutionProvenanceRGCN
    model_config: ProvenanceRgcnModelConfig
    training_config: ProvenanceRgcnTrainingConfig
    metric_records: list[dict[str, float | int]]
    optimizer_state_dict: dict[str, object]
    best_model_state_dict: dict[str, Tensor]
    best_epoch: int
    best_dev_metric: float
    best_metrics: dict[str, float]


@dataclass(frozen=True)
class ProvenanceDevMetrics:
    full_support_at_5: float
    mrr: float
    edge_precision_at_10: float
    edge_recall_at_10: float
    edge_f1_at_10: float
    average_emitted_edges: float
    abstention_rate: float

    @property
    def joint(self) -> float:
        return (
            0.50 * self.full_support_at_5
            + 0.25 * self.mrr
            + 0.25 * self.edge_f1_at_10
        )

    def selection_key(self, epoch: int) -> tuple[float, float, float, float, int]:
        return (
            self.joint,
            self.full_support_at_5,
            self.edge_f1_at_10,
            self.mrr,
            -epoch,
        )

    def to_record(self) -> dict[str, float]:
        return {
            "dev_joint": self.joint,
            "dev_full_support_at_5": self.full_support_at_5,
            "dev_mrr": self.mrr,
            "dev_edge_precision_at_10": self.edge_precision_at_10,
            "dev_edge_recall_at_10": self.edge_recall_at_10,
            "dev_edge_f1_at_10": self.edge_f1_at_10,
            "dev_average_emitted_edges": self.average_emitted_edges,
            "dev_abstention_rate": self.abstention_rate,
        }


def compute_provenance_loss(
    tensor: ProvenanceGraphTensor,
    output: ProvenanceModelOutput,
    label: EvidenceLabel,
    train_pairs: list[TrainPairRecord],
    config: ProvenanceRgcnModelConfig,
    training: ProvenanceRgcnTrainingConfig,
) -> ProvenanceLoss:
    _ = config
    candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(tensor.candidate_ids)
    }
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    seen_pair_nodes: set[str] = set()
    for pair in train_pairs:
        if pair["task_id"] != label.task_id:
            raise ValueError(
                "Provenance train pair task mismatch: "
                f"expected={label.task_id!r} observed={pair['task_id']!r}."
            )
        node_id = pair["node_id"]
        if node_id not in candidate_index:
            raise ValueError(
                f"Provenance train pair node_id={node_id!r} is not a candidate."
            )
        if node_id in seen_pair_nodes:
            raise ValueError(
                "Provenance train pairs must contain each candidate at most once: "
                f"task_id={label.task_id!r} node_id={node_id!r}."
            )
        seen_pair_nodes.add(node_id)
        if pair["label"] == 1:
            positive_indices.append(candidate_index[node_id])
        else:
            negative_indices.append(candidate_index[node_id])
    if not positive_indices:
        raise ValueError(
            f"Provenance task_id={label.task_id!r} has no positive train pairs."
        )
    comparison_count = len(positive_indices) * len(negative_indices)
    if negative_indices:
        margins = (
            output.candidate_logits[positive_indices].unsqueeze(1)
            - output.candidate_logits[negative_indices].unsqueeze(0)
        )
        candidate_loss = F.softplus(-margins).mean()
    else:
        candidate_loss = output.candidate_logits.sum() * 0.0
    gold_edges = set(label.gold_dependency_edges)
    if tensor.logical_transitions:
        edge_targets = output.edge_logits.new_tensor(
            [
                float((item.source_id, item.target_id) in gold_edges)
                for item in tensor.logical_transitions
            ]
        )
        positive_count = int(edge_targets.sum().item())
        negative_count = len(edge_targets) - positive_count
        pos_weight = (
            output.edge_logits.new_tensor(
                float(negative_count) / float(positive_count)
            )
            if positive_count and negative_count
            else None
        )
        edge_loss = F.binary_cross_entropy_with_logits(
            output.edge_logits,
            edge_targets,
            pos_weight=pos_weight,
        )
    else:
        edge_loss = output.candidate_logits.new_zeros(())

    total = (
        training.candidate_loss_weight * candidate_loss
        + training.edge_loss_weight * edge_loss
    )
    return ProvenanceLoss(
        total=total,
        candidate=candidate_loss,
        edge=edge_loss,
        comparison_count=comparison_count,
    )


def train_provenance_rgcn(
    *,
    train_requests: list[ExecutionProvenanceRankingRequest],
    train_labels: list[EvidenceLabel],
    train_pairs: list[TrainPairRecord],
    model_config: ProvenanceRgcnModelConfig,
    training_config: ProvenanceRgcnTrainingConfig,
    encoder: SentenceEncoder,
    dev_requests: list[ExecutionProvenanceRankingRequest] | None = None,
    dev_labels: list[EvidenceLabel] | None = None,
    device: str | torch.device = "cpu",
) -> ProvenanceTrainingResult:
    labels_by_task = {label.task_id: label for label in train_labels}
    if set(labels_by_task) != {request.task_id for request in train_requests}:
        raise ValueError("Provenance train request/label task IDs must align.")
    pairs_by_task: dict[str, list[TrainPairRecord]] = {
        request.task_id: [] for request in train_requests
    }
    for pair in train_pairs:
        if pair["task_id"] not in pairs_by_task:
            raise ValueError(
                "Provenance train pair has no matching request: "
                f"task_id={pair['task_id']!r}."
            )
        pairs_by_task[pair["task_id"]].append(pair)
    torch.manual_seed(training_config.random_seed)
    target_device = torch.device(device)
    model = ExecutionProvenanceRGCN(model_config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    tensors = [
        tensorize_provenance_request(request, encoder=encoder, config=model_config)
        for request in train_requests
    ]
    metrics: list[dict[str, float | int]] = []
    best_metric = float("-inf")
    best_dev_metrics: ProvenanceDevMetrics | None = None
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_total = 0.0
        candidate_loss_total = 0.0
        edge_loss_total = 0.0
        comparison_count = 0
        for index, (request, tensor) in enumerate(
            zip(train_requests, tensors, strict=True), start=1
        ):
            moved = move_provenance_tensor(tensor, target_device)
            loss = compute_provenance_loss(
                moved,
                model(moved),
                labels_by_task[request.task_id],
                pairs_by_task[request.task_id],
                model_config,
                training_config,
            )
            (loss.total / training_config.batch_size).backward()
            loss_total += float(loss.total.detach().cpu())
            candidate_loss_total += float(loss.candidate.detach().cpu())
            edge_loss_total += float(loss.edge.detach().cpu())
            comparison_count += loss.comparison_count
            if index % training_config.batch_size == 0 or index == len(tensors):
                nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        dev_metrics = _dev_metrics(
            model,
            requests=dev_requests or train_requests,
            labels=dev_labels or train_labels,
            encoder=encoder,
            config=model_config,
            device=target_device,
        )
        if best_dev_metrics is None or dev_metrics.selection_key(
            epoch
        ) > best_dev_metrics.selection_key(best_epoch):
            best_metric = dev_metrics.joint
            best_dev_metrics = dev_metrics
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": loss_total / max(1, len(tensors)),
            "train_candidate_loss": candidate_loss_total / max(1, len(tensors)),
            "train_edge_loss": edge_loss_total / max(1, len(tensors)),
            "train_task_count": len(tensors),
            "train_comparison_count": comparison_count,
            **dev_metrics.to_record(),
        }
        pair_category_counts = {
            sample_type: sum(
                pair["sample_type"] == sample_type for pair in train_pairs
            )
            for sample_type in sorted({pair["sample_type"] for pair in train_pairs})
        }
        epoch_metrics.update(
            {
                f"train_pair_{sample_type}_count": count
                for sample_type, count in pair_category_counts.items()
            }
        )
        metrics.append(epoch_metrics)
    model.load_state_dict(best_state)
    if best_dev_metrics is None:
        raise RuntimeError("Provenance training did not evaluate any epoch.")
    return ProvenanceTrainingResult(
        model=model,
        model_config=model_config,
        training_config=training_config,
        metric_records=metrics,
        optimizer_state_dict=optimizer.state_dict(),
        best_model_state_dict=best_state,
        best_epoch=best_epoch,
        best_dev_metric=best_metric,
        best_metrics=best_dev_metrics.to_record(),
    )


def _dev_metrics(
    model: ExecutionProvenanceRGCN,
    *,
    requests: list[ExecutionProvenanceRankingRequest],
    labels: list[EvidenceLabel],
    encoder: SentenceEncoder,
    config: ProvenanceRgcnModelConfig,
    device: torch.device,
) -> ProvenanceDevMetrics:
    labels_by_task = {label.task_id: label for label in labels}
    complete = 0
    reciprocal_rank_total = 0.0
    edge_true_positive_count = 0
    predicted_edge_count = 0
    gold_edge_count = 0
    emitted_edge_count = 0
    abstained_source_count = 0
    considered_source_count = 0
    retriever = ExecutionProvenanceRgcnRetriever(
        model=model,
        encoder=encoder,
        config=config,
        device=device,
    )
    for request in requests:
        result = retriever.rank_task(request, top_k=10)
        ranked_ids = [item.node_id for item in result.ranked_nodes]
        label = labels_by_task[request.task_id]
        gold_ids = set(label.gold_evidence_item_ids)
        complete += int(gold_ids <= set(ranked_ids[:5]))
        reciprocal_rank_total += next(
            (
                1.0 / rank
                for rank, node_id in enumerate(ranked_ids, start=1)
                if node_id in gold_ids
            ),
            0.0,
        )
        predicted_edges = {
            (edge["source"], edge["target"])
            for edge in result.trace.retrieved_edges
        }
        gold_edges = set(label.gold_dependency_edges)
        edge_true_positive_count += len(predicted_edges & gold_edges)
        predicted_edge_count += len(predicted_edges)
        gold_edge_count += len(gold_edges)
        emitted_edge_count += len(predicted_edges)
        native_trace = result.trace.native_trace
        if isinstance(native_trace, ExecutionProvenanceTrace):
            considered_sources = {
                item.source_id for item in native_trace.structured_transitions
            }
            considered_source_count += len(considered_sources)
            abstained_source_count += len(
                set(native_trace.abstained_source_ids) & considered_sources
            )
    task_count = max(1, len(requests))
    edge_precision = (
        edge_true_positive_count / predicted_edge_count if predicted_edge_count else 0.0
    )
    edge_recall = (
        edge_true_positive_count / gold_edge_count if gold_edge_count else 0.0
    )
    edge_f1 = (
        2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)
        if edge_precision + edge_recall
        else 0.0
    )
    return ProvenanceDevMetrics(
        full_support_at_5=complete / task_count,
        mrr=reciprocal_rank_total / task_count,
        edge_precision_at_10=edge_precision,
        edge_recall_at_10=edge_recall,
        edge_f1_at_10=edge_f1,
        average_emitted_edges=emitted_edge_count / task_count,
        abstention_rate=(
            abstained_source_count / considered_source_count
            if considered_source_count
            else 0.0
        ),
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in deepcopy(model.state_dict()).items()
    }


__all__ = [
    "ProvenanceLoss",
    "ProvenanceDevMetrics",
    "ProvenanceTrainingResult",
    "compute_provenance_loss",
    "train_provenance_rgcn",
]
