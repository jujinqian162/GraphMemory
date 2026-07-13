from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_retriever.config.records import (
    BeamLossConfig,
    BeamSearchConfig,
)
from graph_memory.models.graph_retriever.decoder import (
    BeamHypothesis,
    joint_action_log_probabilities,
    normalize_beam_score,
    prune_beam_hypotheses,
)
from graph_memory.models.graph_retriever.internals.contracts import (
    EncodedGraphState,
    TrainingBatch,
)
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringModel
from graph_memory.models.graph_retriever.oracle import DynamicEvidenceOracle


@dataclass(frozen=True)
class SupervisedActionLosses:
    next_action_loss: Tensor
    stop_loss: Tensor
    stop_probability: Tensor


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


def supervised_action_losses(
    *,
    candidate_logits: Tensor,
    stop_logit: Tensor,
    eligible_candidate_indices: Sequence[int],
    valid_candidate_indices: Sequence[int],
    stop_target: bool,
) -> SupervisedActionLosses:
    eligible = tuple(eligible_candidate_indices)
    valid = tuple(valid_candidate_indices)
    eligible_set = set(eligible)
    if any(index not in eligible_set for index in valid):
        raise ValueError("valid candidate actions must be eligible")
    if stop_target and valid:
        raise ValueError("STOP must be the only valid action after gold completion")
    if not stop_target and not valid:
        raise ValueError("incomplete evidence requires a valid candidate action")

    joint_log_probs = joint_action_log_probabilities(
        candidate_logits[list(eligible)], stop_logit
    )
    stop_log_probability = joint_log_probs[-1]
    if stop_target:
        valid_log_probability = stop_log_probability
        stop_loss = -stop_log_probability
    else:
        eligible_positions = {
            index: position for position, index in enumerate(eligible)
        }
        valid_positions = [eligible_positions[index] for index in valid]
        valid_log_probability = torch.logsumexp(joint_log_probs[valid_positions], dim=0)
        candidate_log_probability = torch.logsumexp(joint_log_probs[:-1], dim=0)
        stop_loss = -candidate_log_probability
    return SupervisedActionLosses(
        next_action_loss=-valid_log_probability,
        stop_loss=stop_loss,
        stop_probability=stop_log_probability.exp(),
    )


def compute_beam_loss(
    *,
    model: EvidenceScoringModel,
    batch: TrainingBatch,
    labels: list[EvidenceLabel],
    loss_config: BeamLossConfig,
    beam_search_config: BeamSearchConfig,
    pos_weight: Tensor | None = None,
    encoded: EncodedGraphState | None = None,
) -> BeamLossBreakdown:
    encoded_state = encoded if encoded is not None else model.encode_graph(batch)
    auxiliary_logits = model.base_score(
        node_states=encoded_state.node_states[batch.sample_node_indices],
        query_states=encoded_state.node_states[batch.sample_query_indices],
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

    for task_index, task_id in enumerate(encoded_state.task_ids):
        if task_id not in labels_by_task:
            raise ValueError(
                f"Beam training is missing EvidenceLabel for task_id={task_id}"
            )
        candidate_ids = tuple(encoded_state.candidate_node_ids_by_task[task_index])
        oracle = DynamicEvidenceOracle(
            label=labels_by_task[task_id],
            candidate_node_ids=candidate_ids,
            max_steps=beam_search_config.max_steps,
        )
        beam = [BeamHypothesis()]
        for _ in range(beam_search_config.max_steps):
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
                    encoded_state,
                    task_index=task_index,
                    selected_local_indices=hypothesis.selected_indices,
                )
                eligible = tuple(
                    index
                    for index, node_id in enumerate(candidate_ids)
                    if index not in hypothesis.selected_set
                    and node_id not in oracle_state.masked_future_ids
                )
                valid = tuple(
                    index
                    for index in eligible
                    if candidate_ids[index] in oracle_state.valid_next_ids
                )
                supervised = supervised_action_losses(
                    candidate_logits=scores.candidate_logits,
                    stop_logit=scores.stop_logit,
                    eligible_candidate_indices=eligible,
                    valid_candidate_indices=valid,
                    stop_target=oracle_state.stop_target,
                )
                action_losses.append(supervised.next_action_loss)
                stop_losses.append(supervised.stop_loss)

                raw_available = tuple(
                    index
                    for index in range(len(candidate_ids))
                    if index not in hypothesis.selected_set
                )
                log_probs = joint_action_log_probabilities(
                    scores.candidate_logits[list(raw_available)], scores.stop_logit
                )
                cpu_log_probs = log_probs.detach().cpu().tolist()
                stop_logit_value = float(scores.stop_logit.detach().cpu())
                for position, candidate_index in enumerate(raw_available):
                    raw_score = hypothesis.raw_score + float(cpu_log_probs[position])
                    selected = (*hypothesis.selected_indices, candidate_index)
                    candidate = BeamHypothesis(
                        selected_indices=selected,
                        raw_score=raw_score,
                        normalized_score=normalize_beam_score(
                            raw_score,
                            len(selected),
                            beam_search_config.length_penalty_alpha,
                        ),
                    )
                    expansions.append(candidate)
                    selected_node_ids = tuple(
                        candidate_ids[index] for index in selected
                    )
                    if oracle.state(selected_node_ids).oracle_reachable:
                        reachable_expansions.append(candidate)
                stop_raw_score = hypothesis.raw_score + float(cpu_log_probs[-1])
                stop_candidate = BeamHypothesis(
                    selected_indices=hypothesis.selected_indices,
                    raw_score=stop_raw_score,
                    normalized_score=normalize_beam_score(
                        stop_raw_score,
                        len(hypothesis.selected_indices) + 1,
                        beam_search_config.length_penalty_alpha,
                    ),
                    stopped=True,
                    stop_score=stop_logit_value,
                )
                expansions.append(stop_candidate)
                stop_decisions += 1
                if not oracle_state.stop_target:
                    premature_stops += int(
                        float(supervised.stop_probability.detach().cpu()) >= 0.5
                    )
            beam = prune_beam_hypotheses(
                expansions,
                candidate_node_ids=candidate_ids,
                beam_size=beam_search_config.training_beam_size,
                deduplicate=beam_search_config.deduplicate_selected_sets,
            )
            if reachable_expansions and not any(
                oracle.state(
                    tuple(candidate_ids[index] for index in item.selected_indices)
                ).oracle_reachable
                and not item.stopped
                for item in beam
            ):
                replacement = prune_beam_hypotheses(
                    reachable_expansions,
                    candidate_node_ids=candidate_ids,
                    beam_size=beam_search_config.training_beam_size,
                    deduplicate=beam_search_config.deduplicate_selected_sets,
                )[0]
                beam = prune_beam_hypotheses(
                    [*beam[:-1], replacement],
                    candidate_node_ids=candidate_ids,
                    beam_size=beam_search_config.training_beam_size,
                    deduplicate=beam_search_config.deduplicate_selected_sets,
                )
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
    total = (
        loss_config.next_action_loss_weight * next_action_loss
        + loss_config.stop_loss_weight * stop_loss
        + loss_config.aux_node_loss_weight * aux_loss
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


__all__ = [
    "BeamLossBreakdown",
    "SupervisedActionLosses",
    "compute_beam_loss",
    "supervised_action_losses",
]
