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
from graph_memory.models.provenance_rgcn.tensorization import (
    move_provenance_tensor,
    tensorize_provenance_request,
)
from graph_memory.retrieval.requests import ExecutionProvenanceRankingRequest


@dataclass(frozen=True)
class ProvenanceLoss:
    total: Tensor
    candidate: Tensor
    edge: Tensor


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
    pair_indices: list[int] = []
    pair_targets: list[float] = []
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
        pair_indices.append(candidate_index[node_id])
        pair_targets.append(float(pair["label"]))
    if not pair_indices:
        raise ValueError(
            f"Provenance task_id={label.task_id!r} has no materialized train pairs."
        )
    candidate_loss = F.binary_cross_entropy_with_logits(
        output.candidate_logits[pair_indices],
        output.candidate_logits.new_tensor(pair_targets),
    )
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
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    for epoch in range(1, training_config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_total = 0.0
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
            if index % training_config.batch_size == 0 or index == len(tensors):
                nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
        dev_metric = _dev_full_support(
            model,
            requests=dev_requests or train_requests,
            labels=dev_labels or train_labels,
            encoder=encoder,
            config=model_config,
            device=target_device,
        )
        if dev_metric > best_metric:
            best_metric = dev_metric
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / max(1, len(tensors)),
                "dev_full_support_at_5": dev_metric,
            }
        )
    model.load_state_dict(best_state)
    return ProvenanceTrainingResult(
        model=model,
        model_config=model_config,
        training_config=training_config,
        metric_records=metrics,
        optimizer_state_dict=optimizer.state_dict(),
        best_model_state_dict=best_state,
        best_epoch=best_epoch,
        best_dev_metric=best_metric,
    )


def _dev_full_support(
    model: ExecutionProvenanceRGCN,
    *,
    requests: list[ExecutionProvenanceRankingRequest],
    labels: list[EvidenceLabel],
    encoder: SentenceEncoder,
    config: ProvenanceRgcnModelConfig,
    device: torch.device,
) -> float:
    labels_by_task = {label.task_id: label for label in labels}
    complete = 0
    model.eval()
    with torch.no_grad():
        for request in requests:
            tensor = move_provenance_tensor(
                tensorize_provenance_request(request, encoder=encoder, config=config),
                device,
            )
            output = model(tensor)
            order = sorted(
                range(len(tensor.candidate_ids)),
                key=lambda index: (
                    -float(output.candidate_logits[index].detach().cpu()),
                    tensor.candidate_ids[index],
                ),
            )
            selected = {tensor.candidate_ids[index] for index in order[:5]}
            complete += int(
                set(labels_by_task[request.task_id].gold_evidence_item_ids) <= selected
            )
    return complete / max(1, len(requests))


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in deepcopy(model.state_dict()).items()
    }


__all__ = [
    "ProvenanceLoss",
    "ProvenanceTrainingResult",
    "compute_provenance_loss",
    "train_provenance_rgcn",
]
