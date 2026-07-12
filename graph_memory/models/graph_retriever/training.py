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
from graph_memory.models.graph_retriever.config.records import (
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.dev_evaluation import (
    best_metric as select_best_metric,
)
from graph_memory.models.graph_retriever.dev_evaluation import predict_dev_from_batches
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.models.graph_retriever.decoder import BeamHypothesis
from graph_memory.models.graph_retriever.oracle import DynamicEvidenceOracle
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


@dataclass(frozen=True)
class BeamLossBreakdown:
    total_loss: Tensor
    next_action_loss: Tensor
    stop_loss: Tensor
    aux_node_loss: Tensor
    retained_hypotheses: float
    oracle_reachable_rate: float
    premature_stop_rate: float
    average_selected_length: float


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
    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if _is_decoder_parameter(name)
    ]
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not _is_decoder_parameter(name)
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": training_config.optimizer_phase_config.rgcn_learning_rate,
                "name": "rgcn",
            },
            {
                "params": decoder_parameters,
                "lr": training_config.optimizer_phase_config.decoder_learning_rate,
                "name": "decoder",
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    train_batches = build_training_batches(
        ranking_requests=train_requests,
        graphs=train_graphs,
        pairs=train_pairs,
        model_config=model_config,
        text_embedding_provider=text_embedding_provider,
        seed_signal_provider=seed_signal_provider,
        batch_size=training_config.batch_size,
        labels=train_labels,
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

    pos_weight = (
        _pos_weight(train_pairs, device) if training_config.pos_weight_enabled else None
    )
    metric_records: list[MetricRecord] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    global_step = 0
    negative_count_by_type = _negative_count_by_type(train_pairs)
    positive_count = sum(1 for pair in train_pairs if pair["label"] == 1)

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        encoder_lr = (
            0.0
            if epoch <= training_config.optimizer_phase_config.decoder_warmup_epochs
            else training_config.optimizer_phase_config.rgcn_learning_rate
        )
        optimizer.param_groups[0]["lr"] = encoder_lr
        train_loss_total = 0.0
        train_sample_count = 0
        last_grad_norm = 0.0
        component_totals = {
            "next_action_loss": 0.0,
            "stop_loss": 0.0,
            "aux_node_loss": 0.0,
            "retained_hypotheses": 0.0,
            "oracle_reachable_rate": 0.0,
            "premature_stop_rate": 0.0,
            "average_selected_length": 0.0,
        }
        for batch in train_batches:
            moved_batch = move_training_batch(batch, device)
            labels_for_batch = _labels_for_training_batch(
                moved_batch, train_labels, train_pairs
            )
            breakdown = compute_beam_training_loss(
                model=model,
                batch=moved_batch,
                labels=labels_for_batch,
                training_config=training_config,
                pos_weight=pos_weight,
            )
            loss = breakdown.total_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), training_config.max_grad_norm
            )
            optimizer.step()
            global_step += 1
            sample_count = int(moved_batch.labels.shape[0])
            train_loss_total += float(loss.detach().cpu()) * sample_count
            train_sample_count += sample_count
            last_grad_norm = float(
                grad_norm.detach().cpu() if isinstance(grad_norm, Tensor) else grad_norm
            )
            for key in component_totals:
                value = getattr(breakdown, key)
                component_totals[key] += (
                    float(value.detach().cpu())
                    if isinstance(value, Tensor)
                    else float(value)
                )

        dev_predictions, dev_loss = predict_dev_from_batches(
            model=model,
            ranking_requests=dev_requests,
            labels=dev_labels,
            graphs=dev_graphs,
            model_config=model_config,
            batches=dev_batches,
            device=device,
        )
        dev_rows = evaluate_results(
            EvidenceEvaluationRequest(
                predictions=dev_predictions, labels=dev_labels, graphs=dev_graphs
            )
        )
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
                "train_loss": train_loss_total / train_sample_count
                if train_sample_count
                else 0.0,
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
                **{
                    key: value / len(train_batches)
                    for key, value in component_totals.items()
                },
                "beam_size": model_config.beam_search_config.training_beam_size,
                "max_steps": model_config.beam_search_config.max_steps,
                "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "decoder_learning_rate": float(optimizer.param_groups[1]["lr"]),
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


def compute_beam_training_loss(
    *,
    model: EvidenceScoringModel,
    batch: TrainingBatch,
    labels: list[EvidenceLabel],
    training_config: RgcnTrainingConfig,
    pos_weight: Tensor | None = None,
) -> BeamLossBreakdown:
    encoded = model.encode_graph(batch)
    auxiliary_logits = model.base_score(
        node_states=encoded.node_states[batch.sample_node_indices],
        query_states=encoded.node_states[batch.sample_query_indices],
        sample_node_features=batch.sample_node_features,
    )
    aux_loss = F.binary_cross_entropy_with_logits(
        auxiliary_logits, batch.labels, pos_weight=pos_weight
    )
    labels_by_task = {label.task_id: label for label in labels}
    action_losses: list[Tensor] = []
    stop_losses: list[Tensor] = []
    retained_counts: list[int] = []
    reachable_counts = 0
    state_counts = 0
    premature_stops = 0
    stop_decisions = 0
    selected_lengths: list[int] = []

    for task_index, task_id in enumerate(encoded.task_ids):
        if task_id not in labels_by_task:
            raise ValueError(
                f"Beam training is missing EvidenceLabel for task_id={task_id}"
            )
        candidate_ids = tuple(encoded.candidate_node_ids_by_task[task_index])
        oracle = DynamicEvidenceOracle(
            label=labels_by_task[task_id],
            candidate_node_ids=candidate_ids,
            max_steps=model_config_max_steps(model),
        )
        beam = [BeamHypothesis()]
        for _ in range(model_config_max_steps(model)):
            expansions: list[BeamHypothesis] = []
            reachable_expansions: list[BeamHypothesis] = []
            for hypothesis in beam:
                if hypothesis.stopped:
                    expansions.append(hypothesis)
                    continue
                selected_ids = tuple(
                    candidate_ids[index] for index in hypothesis.selected_indices
                )
                oracle_state = oracle.state(selected_ids)
                scores = model.score_decoder_actions(
                    encoded,
                    task_index=task_index,
                    selected_local_indices=hypothesis.selected_indices,
                )
                stop_target = torch.tensor(
                    float(oracle_state.stop_target),
                    dtype=scores.stop_logit.dtype,
                    device=scores.stop_logit.device,
                )
                stop_losses.append(
                    F.binary_cross_entropy_with_logits(scores.stop_logit, stop_target)
                )
                available = [
                    index
                    for index, node_id in enumerate(candidate_ids)
                    if index not in hypothesis.selected_set
                    and node_id not in oracle_state.masked_future_ids
                ]
                valid = [
                    index
                    for index in available
                    if candidate_ids[index] in oracle_state.valid_next_ids
                ]
                if valid:
                    action_losses.append(
                        torch.logsumexp(scores.candidate_logits[available], dim=0)
                        - torch.logsumexp(scores.candidate_logits[valid], dim=0)
                    )
                raw_available = [
                    index
                    for index in range(len(candidate_ids))
                    if index not in hypothesis.selected_set
                ]
                logits = torch.cat(
                    [
                        scores.candidate_logits[raw_available],
                        scores.stop_logit.reshape(1),
                    ]
                )
                cpu_values = (
                    torch.cat(
                        [torch.log_softmax(logits, dim=0), scores.stop_logit.reshape(1)]
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                log_probs = cpu_values[:-1]
                stop_logit_value = float(cpu_values[-1])
                for position, candidate_index in enumerate(raw_available):
                    raw_score = hypothesis.raw_score + float(log_probs[position])
                    selected = (*hypothesis.selected_indices, candidate_index)
                    candidate = BeamHypothesis(
                        selected_indices=selected,
                        raw_score=raw_score,
                        normalized_score=_normalized_training_score(
                            raw_score, len(selected), model
                        ),
                    )
                    expansions.append(candidate)
                    selected_node_ids = tuple(
                        candidate_ids[index] for index in selected
                    )
                    if oracle.state(selected_node_ids).oracle_reachable:
                        reachable_expansions.append(candidate)
                stop_candidate = BeamHypothesis(
                    selected_indices=hypothesis.selected_indices,
                    raw_score=hypothesis.raw_score + float(log_probs[-1]),
                    normalized_score=_normalized_training_score(
                        hypothesis.raw_score + float(log_probs[-1]),
                        max(1, len(hypothesis.selected_indices)),
                        model,
                    ),
                    stopped=True,
                    stop_score=stop_logit_value,
                )
                expansions.append(stop_candidate)
                stop_decisions += 1
                if not oracle_state.stop_target:
                    premature_stops += int(stop_logit_value >= 0.0)
            beam = _training_prune(expansions, candidate_ids, model)
            if reachable_expansions and not any(
                oracle.state(
                    tuple(candidate_ids[index] for index in item.selected_indices)
                ).oracle_reachable
                and not item.stopped
                for item in beam
            ):
                replacement = _training_prune(
                    reachable_expansions, candidate_ids, model
                )[0]
                beam = _training_prune([*beam[:-1], replacement], candidate_ids, model)
            retained_counts.append(len(beam))
            state_counts += len(beam)
            reachable_counts += sum(
                oracle.state(
                    tuple(candidate_ids[index] for index in item.selected_indices)
                ).oracle_reachable
                and not item.stopped
                for item in beam
            )
            if all(item.stopped for item in beam):
                break
        selected_lengths.extend(len(item.selected_indices) for item in beam)

    zero = aux_loss * 0.0
    next_action_loss = torch.stack(action_losses).mean() if action_losses else zero
    stop_loss = torch.stack(stop_losses).mean() if stop_losses else zero
    weights = training_config.beam_loss_config
    total = (
        weights.next_action_loss_weight * next_action_loss
        + weights.stop_loss_weight * stop_loss
        + weights.aux_node_loss_weight * aux_loss
    )
    return BeamLossBreakdown(
        total_loss=total,
        next_action_loss=next_action_loss,
        stop_loss=stop_loss,
        aux_node_loss=aux_loss,
        retained_hypotheses=(
            sum(retained_counts) / len(retained_counts) if retained_counts else 0.0
        ),
        oracle_reachable_rate=(
            reachable_counts / state_counts if state_counts else 0.0
        ),
        premature_stop_rate=(
            premature_stops / stop_decisions if stop_decisions else 0.0
        ),
        average_selected_length=(
            sum(selected_lengths) / len(selected_lengths) if selected_lengths else 0.0
        ),
    )


def model_config_max_steps(model: EvidenceScoringModel) -> int:
    return model.max_steps


def _normalized_training_score(
    raw_score: float, length: int, model: EvidenceScoringModel
) -> float:
    alpha = getattr(model, "length_penalty_alpha", 1.0)
    return raw_score / (max(1, length) ** alpha)


def _training_prune(
    hypotheses: list[BeamHypothesis],
    candidate_ids: tuple[str, ...],
    model: EvidenceScoringModel,
) -> list[BeamHypothesis]:
    beam_size = getattr(model, "training_beam_size", 2)
    ordered = sorted(
        hypotheses,
        key=lambda item: (
            -item.normalized_score,
            -item.raw_score,
            tuple(candidate_ids[index] for index in item.selected_indices),
            not item.stopped,
        ),
    )
    result: list[BeamHypothesis] = []
    seen: set[frozenset[int]] = set()
    for hypothesis in ordered:
        if hypothesis.selected_set in seen:
            continue
        seen.add(hypothesis.selected_set)
        result.append(hypothesis)
        if len(result) >= beam_size:
            break
    return result


def _labels_for_training_batch(
    batch: TrainingBatch,
    labels: list[EvidenceLabel] | None,
    pairs: list[TrainPairRecord],
) -> list[EvidenceLabel]:
    if labels is not None:
        by_task = {label.task_id: label for label in labels}
        return [by_task[task_id] for task_id in batch.graph_batch.task_ids]
    positive_by_task: dict[str, list[str]] = {}
    for pair in pairs:
        if pair["label"] == 1:
            positive_by_task.setdefault(pair["task_id"], []).append(pair["node_id"])
    return [
        EvidenceLabel(
            task_id=task_id,
            gold_answer="",
            gold_evidence_item_ids=tuple(positive_by_task.get(task_id, ())),
            gold_dependency_edges=(),
        )
        for task_id in batch.graph_batch.task_ids
    ]


def _is_decoder_parameter(name: str) -> bool:
    return name.startswith(
        (
            "first_hop_scorer",
            "subsequent_hop_scorer",
            "stop_scorer",
            "step_embedding",
            "frontier_projection",
        )
    )
