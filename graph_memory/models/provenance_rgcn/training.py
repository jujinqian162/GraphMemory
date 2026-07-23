from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.embeddings import SentenceEncoder
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_batching import TaskTensorDataset
from graph_memory.models.provenance_rgcn.config import (
    ProvenanceRgcnModelConfig,
    ProvenanceRgcnTrainingConfig,
)
from graph_memory.models.provenance_rgcn.contracts import (
    ProvenanceGraphTensor,
    ProvenanceModelOutput,
    ProvenanceTaskTensor,
    ProvenanceTrainingBatch,
    ProvenanceTrainingTask,
)
from graph_memory.models.provenance_rgcn.inference import (
    rank_provenance_task_output,
)
from graph_memory.models.provenance_rgcn.model import ExecutionProvenanceRGCN
from graph_memory.models.provenance_rgcn.tensorization import (
    collate_provenance_tasks,
    collate_provenance_training_tasks,
    materialize_provenance_training_task,
    move_provenance_tensor,
    move_provenance_training_batch,
    split_provenance_output,
    tensorize_provenance_task,
)
from graph_memory.retrieval.contracts import ExecutionProvenanceTrace
from graph_memory.retrieval.requests import ExecutionProvenanceRankingRequest


@dataclass(frozen=True)
class ProvenanceLoss:
    total: Tensor
    candidate: Tensor
    edge: Tensor
    comparison_count: int
    task_count: int = 1


@dataclass(frozen=True)
class ProvenanceTrainingResult:
    model: ExecutionProvenanceRGCN
    model_config: ProvenanceRgcnModelConfig
    training_config: ProvenanceRgcnTrainingConfig
    metric_records: list[dict[str, object]]
    optimizer_state_dict: dict[str, object]
    best_model_state_dict: dict[str, Tensor]
    best_epoch: int
    global_step: int
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
            0.50 * self.full_support_at_5 + 0.25 * self.mrr + 0.25 * self.edge_f1_at_10
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
    """Compatibility wrapper for v2 loss over a one-task graph batch."""

    if tensor.task_count != 1:
        raise ValueError(
            "compute_provenance_loss accepts one task; use "
            "compute_provenance_batch_loss for multiple tasks."
        )
    candidate_ids = tensor.candidate_ids_by_task[0]
    candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    candidate_targets = torch.full(
        (len(candidate_ids),),
        -1,
        dtype=torch.int8,
        device=output.candidate_logits.device,
    )
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
        candidate_targets[candidate_index[node_id]] = int(pair["label"])
    if not bool((candidate_targets == 1).any()):
        raise ValueError(
            f"Provenance task_id={label.task_id!r} has no positive train pairs."
        )
    gold_edges = set(label.gold_dependency_edges)
    edge_targets = output.edge_logits.new_tensor(
        [
            float((item.source_id, item.target_id) in gold_edges)
            for item in tensor.logical_transitions_by_task[0]
        ]
    )
    return compute_provenance_batch_loss(
        ProvenanceTrainingBatch(
            tensor=tensor,
            candidate_targets=candidate_targets,
            edge_targets=edge_targets,
        ),
        output,
        config,
        training,
    )


def compute_provenance_batch_loss(
    batch: ProvenanceTrainingBatch,
    output: ProvenanceModelOutput,
    config: ProvenanceRgcnModelConfig,
    training: ProvenanceRgcnTrainingConfig,
) -> ProvenanceLoss:
    """Compute task-local v2 candidate/edge losses and an equal-task mean."""

    _ = config
    tensor = batch.tensor
    if len(output.candidate_logits) != len(batch.candidate_targets):
        raise ValueError("Candidate logits and targets must have equal lengths.")
    if len(output.edge_logits) != len(batch.edge_targets):
        raise ValueError("Edge logits and targets must have equal lengths.")
    candidate_losses: list[Tensor] = []
    edge_losses: list[Tensor] = []
    total_losses: list[Tensor] = []
    comparison_count = 0

    for task_index in range(tensor.task_count):
        candidate_start = tensor.candidate_offsets[task_index]
        candidate_end = tensor.candidate_offsets[task_index + 1]
        task_logits = output.candidate_logits[candidate_start:candidate_end]
        task_targets = batch.candidate_targets[candidate_start:candidate_end]
        positive_indices = torch.where(task_targets == 1)[0]
        negative_indices = torch.where(task_targets == 0)[0]
        if not len(positive_indices):
            raise ValueError(
                "Every provenance task must contain a materialized positive pair."
            )
        comparison_count += int(len(positive_indices) * len(negative_indices))
        if len(negative_indices):
            margins = task_logits[positive_indices].unsqueeze(1) - task_logits[
                negative_indices
            ].unsqueeze(0)
            candidate_loss = F.softplus(-margins).mean()
        else:
            candidate_loss = task_logits.sum() * 0.0

        transition_start = tensor.transition_offsets[task_index]
        transition_end = tensor.transition_offsets[task_index + 1]
        edge_logits = output.edge_logits[transition_start:transition_end]
        edge_targets = batch.edge_targets[transition_start:transition_end]
        if len(edge_logits):
            positive_count = int(edge_targets.sum().item())
            negative_count = len(edge_targets) - positive_count
            pos_weight = (
                edge_logits.new_tensor(float(negative_count) / float(positive_count))
                if positive_count and negative_count
                else None
            )
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits,
                edge_targets,
                pos_weight=pos_weight,
            )
        else:
            edge_loss = task_logits.sum() * 0.0
        task_total = (
            training.candidate_loss_weight * candidate_loss
            + training.edge_loss_weight * edge_loss
        )
        candidate_losses.append(candidate_loss)
        edge_losses.append(edge_loss)
        total_losses.append(task_total)

    return ProvenanceLoss(
        total=torch.stack(total_losses).mean(),
        candidate=torch.stack(candidate_losses).mean(),
        edge=torch.stack(edge_losses).mean(),
        comparison_count=comparison_count,
        task_count=tensor.task_count,
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
    train_tasks = [
        materialize_provenance_training_task(
            request,
            labels_by_task[request.task_id],
            pairs_by_task[request.task_id],
            encoder=encoder,
            config=model_config,
        )
        for request in tqdm(
            train_requests,
            desc="provenance-rgcn train tensors",
            unit="task",
        )
    ]
    if not train_tasks:
        raise ValueError("Provenance training requires at least one task tensor.")
    resolved_dev_requests = dev_requests or train_requests
    resolved_dev_labels = dev_labels or train_labels
    dev_tasks = [
        tensorize_provenance_task(request, encoder=encoder, config=model_config)
        for request in tqdm(
            resolved_dev_requests,
            desc="provenance-rgcn dev tensors",
            unit="task",
        )
    ]
    generator = torch.Generator()
    generator.manual_seed(training_config.random_seed)
    train_loader = cast(
        DataLoader[ProvenanceTrainingBatch],
        DataLoader(
            TaskTensorDataset(train_tasks),
            batch_size=training_config.per_device_graph_batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_provenance_training_tasks,
        ),
    )
    dev_loader = cast(
        DataLoader[ProvenanceGraphTensor],
        DataLoader(
            TaskTensorDataset(dev_tasks),
            batch_size=training_config.per_device_graph_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_provenance_tasks,
        ),
    )

    train_nodes_per_task = [
        len(task.tensor.graph_tensor.node_ids) for task in train_tasks
    ]
    train_edges_per_task = [
        int(task.tensor.graph_tensor.edge_index.shape[1]) for task in train_tasks
    ]
    train_candidates_per_task = [len(task.tensor.candidate_ids) for task in train_tasks]
    train_transitions_per_task = [
        len(task.tensor.logical_transitions) for task in train_tasks
    ]
    dev_nodes_per_task = [len(task.graph_tensor.node_ids) for task in dev_tasks]
    dev_edges_per_task = [
        int(task.graph_tensor.edge_index.shape[1]) for task in dev_tasks
    ]
    dev_candidates_per_task = [len(task.candidate_ids) for task in dev_tasks]
    dev_transitions_per_task = [len(task.logical_transitions) for task in dev_tasks]
    train_node_count = sum(train_nodes_per_task)
    train_edge_count = sum(train_edges_per_task)
    train_candidate_count = sum(train_candidates_per_task)
    train_transition_count = sum(train_transitions_per_task)
    dev_node_count = sum(dev_nodes_per_task)
    dev_edge_count = sum(dev_edges_per_task)
    dev_candidate_count = sum(dev_candidates_per_task)
    dev_transition_count = sum(dev_transitions_per_task)
    materialized_cpu_tensor_bytes = sum(
        _provenance_training_task_tensor_bytes(task) for task in train_tasks
    ) + sum(_provenance_task_tensor_bytes(task) for task in dev_tasks)
    metrics: list[dict[str, object]] = []
    best_metric = float("-inf")
    best_dev_metrics: ProvenanceDevMetrics | None = None
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    global_step = 0
    for epoch in tqdm(
        range(1, training_config.epochs + 1),
        desc="provenance-rgcn epochs",
        unit="epoch",
    ):
        model.train()
        _reset_peak_memory(target_device)
        train_started_at = perf_counter()
        loss_total = 0.0
        candidate_loss_total = 0.0
        edge_loss_total = 0.0
        task_count_total = 0
        comparison_count = 0
        optimizer_step_count = 0
        actual_tasks_per_optimizer_step: list[int] = []
        last_grad_norm = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            moved = move_provenance_training_batch(batch, target_device)
            loss = compute_provenance_batch_loss(
                moved,
                model(moved.tensor),
                model_config,
                training_config,
            )
            loss.total.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), training_config.max_grad_norm
            )
            optimizer.step()
            global_step += 1
            optimizer_step_count += 1
            actual_tasks_per_optimizer_step.append(loss.task_count)
            loss_total += float(loss.total.detach().cpu()) * loss.task_count
            candidate_loss_total += (
                float(loss.candidate.detach().cpu()) * loss.task_count
            )
            edge_loss_total += float(loss.edge.detach().cpu()) * loss.task_count
            task_count_total += loss.task_count
            comparison_count += loss.comparison_count
            last_grad_norm = float(
                grad_norm.detach().cpu()
                if isinstance(grad_norm, Tensor)
                else grad_norm
            )
        train_elapsed_seconds = perf_counter() - train_started_at
        train_peak_device_memory_bytes = _peak_memory(target_device)
        _reset_peak_memory(target_device)

        dev_metrics = _dev_metrics(
            model,
            requests=resolved_dev_requests,
            labels=resolved_dev_labels,
            loader=dev_loader,
            config=model_config,
            device=target_device,
        )
        dev_peak_device_memory_bytes = _peak_memory(target_device)
        if best_dev_metrics is None or dev_metrics.selection_key(
            epoch
        ) > best_dev_metrics.selection_key(best_epoch):
            best_metric = dev_metrics.joint
            best_dev_metrics = dev_metrics
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        epoch_metrics: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_optimizer_step_count": optimizer_step_count,
            "per_device_graph_batch_size": (
                training_config.per_device_graph_batch_size
            ),
            "actual_tasks_per_optimizer_step": actual_tasks_per_optimizer_step,
            "train_node_count": train_node_count,
            "train_edge_count": train_edge_count,
            "train_candidate_count": train_candidate_count,
            "train_transition_count": train_transition_count,
            "dev_task_count": len(dev_tasks),
            "dev_node_count": dev_node_count,
            "dev_edge_count": dev_edge_count,
            "dev_candidate_count": dev_candidate_count,
            "dev_transition_count": dev_transition_count,
            "materialized_cpu_tensor_bytes": materialized_cpu_tensor_bytes,
            "train_elapsed_seconds": train_elapsed_seconds,
            "train_tasks_per_second": (
                task_count_total / train_elapsed_seconds
                if train_elapsed_seconds > 0.0
                else 0.0
            ),
            "train_peak_device_memory_bytes": train_peak_device_memory_bytes,
            "dev_peak_device_memory_bytes": dev_peak_device_memory_bytes,
            **_count_stats("train_nodes_per_task", train_nodes_per_task),
            **_count_stats("train_edges_per_task", train_edges_per_task),
            **_count_stats("train_candidates_per_task", train_candidates_per_task),
            **_count_stats("train_transitions_per_task", train_transitions_per_task),
            **_count_stats("dev_nodes_per_task", dev_nodes_per_task),
            **_count_stats("dev_edges_per_task", dev_edges_per_task),
            **_count_stats("dev_candidates_per_task", dev_candidates_per_task),
            **_count_stats("dev_transitions_per_task", dev_transitions_per_task),
            "train_total_loss": loss_total / max(1, task_count_total),
            "train_candidate_loss": candidate_loss_total / max(1, task_count_total),
            "train_edge_loss": edge_loss_total / max(1, task_count_total),
            "train_task_count": task_count_total,
            "train_comparison_count": comparison_count,
            "grad_norm": last_grad_norm,
            **dev_metrics.to_record(),
        }
        pair_category_counts = {
            sample_type: sum(pair["sample_type"] == sample_type for pair in train_pairs)
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
        global_step=global_step,
        best_dev_metric=best_metric,
        best_metrics=best_dev_metrics.to_record(),
    )


def _dev_metrics(
    model: ExecutionProvenanceRGCN,
    *,
    requests: list[ExecutionProvenanceRankingRequest],
    labels: list[EvidenceLabel],
    loader: DataLoader[ProvenanceGraphTensor],
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
    request_index = 0

    model.eval()
    with torch.no_grad():
        for tensor in loader:
            moved = move_provenance_tensor(tensor, device)
            outputs = split_provenance_output(moved, model(moved))
            for task_index, output in enumerate(outputs):
                request = requests[request_index]
                observed_task_id = tensor.graph_batch.task_ids[task_index]
                if observed_task_id != request.task_id:
                    raise ValueError("Provenance dev DataLoader changed task order.")
                result = rank_provenance_task_output(
                    candidate_ids=tensor.candidate_ids_by_task[task_index],
                    transitions=tensor.logical_transitions_by_task[task_index],
                    output=output,
                    config=config,
                    top_k=10,
                )
                label = labels_by_task[request.task_id]
                ranked_ids = [item.node_id for item in result.ranked_nodes]
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
                request_index += 1
    if request_index != len(requests):
        raise ValueError("Provenance dev DataLoader did not emit every task.")

    task_count = max(1, len(requests))
    edge_precision = (
        edge_true_positive_count / predicted_edge_count if predicted_edge_count else 0.0
    )
    edge_recall = edge_true_positive_count / gold_edge_count if gold_edge_count else 0.0
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


def _count_stats(prefix: str, values: list[int]) -> dict[str, float | int]:
    if not values:
        return {f"{prefix}_min": 0, f"{prefix}_max": 0, f"{prefix}_mean": 0.0}
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_mean": sum(values) / len(values),
    }


def _provenance_task_tensor_bytes(task: ProvenanceTaskTensor) -> int:
    graph = task.graph_tensor
    tensors = (
        graph.node_embeddings,
        graph.node_features,
        graph.edge_index,
        graph.relation_ids,
        graph.edge_weights,
        task.node_type_ids,
        task.candidate_node_indices,
    )
    return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)


def _provenance_training_task_tensor_bytes(task: ProvenanceTrainingTask) -> int:
    return (
        _provenance_task_tensor_bytes(task.tensor)
        + int(task.candidate_targets.numel() * task.candidate_targets.element_size())
        + int(task.edge_targets.numel() * task.edge_targets.element_size())
    )


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in deepcopy(model.state_dict()).items()
    }


__all__ = [
    "ProvenanceDevMetrics",
    "ProvenanceLoss",
    "ProvenanceTrainingResult",
    "compute_provenance_batch_loss",
    "compute_provenance_loss",
    "train_provenance_rgcn",
]
