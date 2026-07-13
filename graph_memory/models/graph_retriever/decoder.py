from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from graph_memory.models.graph_retriever.config.records import BeamSearchConfig


ActionScorer = Callable[[tuple[int, ...]], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class BeamHypothesis:
    selected_indices: tuple[int, ...] = ()
    raw_score: float = 0.0
    normalized_score: float = 0.0
    stopped: bool = False
    stop_score: float | None = None

    @property
    def selected_set(self) -> frozenset[int]:
        return frozenset(self.selected_indices)

    @property
    def last_node(self) -> int | None:
        return self.selected_indices[-1] if self.selected_indices else None

    @property
    def state_key(self) -> tuple[frozenset[int], int | None]:
        return self.selected_set, self.last_node

    @property
    def action_count(self) -> int:
        return len(self.selected_indices) + int(self.stopped)


@dataclass(frozen=True)
class BeamSearchResult:
    winner: BeamHypothesis
    final_beam: tuple[BeamHypothesis, ...]
    steps: int


def run_beam_search(
    *,
    candidate_node_ids: tuple[str, ...],
    score_actions: ActionScorer,
    config: BeamSearchConfig,
    initial_hypotheses: tuple[BeamHypothesis, ...] | None = None,
) -> BeamSearchResult:
    _validate_config(config)
    beam = list(initial_hypotheses or (BeamHypothesis(),))
    steps = 0
    for _ in range(config.max_steps):
        if all(hypothesis.stopped for hypothesis in beam):
            break
        expanded: list[BeamHypothesis] = []
        for hypothesis in beam:
            if hypothesis.stopped:
                expanded.append(hypothesis)
                continue
            candidate_logits, stop_logit = score_actions(hypothesis.selected_indices)
            if candidate_logits.ndim != 1 or candidate_logits.shape[0] != len(
                candidate_node_ids
            ):
                raise ValueError(
                    "score_actions must return one candidate logit per candidate node"
                )
            available = [
                index
                for index in range(len(candidate_node_ids))
                if index not in hypothesis.selected_set
            ]
            action_log_probs = joint_action_log_probabilities(
                candidate_logits[available], stop_logit
            )
            cpu_log_probs = action_log_probs.detach().cpu().tolist()
            stop_score = float(stop_logit.detach().cpu())
            for position, candidate_index in enumerate(available):
                raw_score = hypothesis.raw_score + float(cpu_log_probs[position])
                selected = (*hypothesis.selected_indices, candidate_index)
                expanded.append(
                    BeamHypothesis(
                        selected_indices=selected,
                        raw_score=raw_score,
                        normalized_score=normalize_beam_score(
                            raw_score, len(selected), config.length_penalty_alpha
                        ),
                    )
                )
            stop_raw_score = hypothesis.raw_score + float(cpu_log_probs[-1])
            stopped_action_count = len(hypothesis.selected_indices) + 1
            expanded.append(
                BeamHypothesis(
                    selected_indices=hypothesis.selected_indices,
                    raw_score=stop_raw_score,
                    normalized_score=normalize_beam_score(
                        stop_raw_score,
                        stopped_action_count,
                        config.length_penalty_alpha,
                    ),
                    stopped=True,
                    stop_score=stop_score,
                )
            )
        beam = prune_beam_hypotheses(
            expanded,
            candidate_node_ids=candidate_node_ids,
            beam_size=config.inference_beam_size,
            deduplicate=config.deduplicate_selected_sets,
        )
        steps += 1
    final_beam = tuple(
        sorted(beam, key=lambda item: _sort_key(item, candidate_node_ids))
    )
    return BeamSearchResult(winner=final_beam[0], final_beam=final_beam, steps=steps)


def complete_beam_ranking(
    *,
    candidate_node_ids: tuple[str, ...],
    base_logits: Tensor,
    winner: BeamHypothesis,
    fill_to: int = 5,
) -> tuple[str, ...]:
    if base_logits.ndim != 1 or base_logits.shape[0] != len(candidate_node_ids):
        raise ValueError("base_logits must align with candidate_node_ids")
    selected = [candidate_node_ids[index] for index in winner.selected_indices]
    selected_set = set(selected)
    base_values = base_logits.detach().cpu().tolist()
    residual = sorted(
        (
            (candidate_node_ids[index], float(base_values[index]))
            for index in range(len(candidate_node_ids))
            if candidate_node_ids[index] not in selected_set
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if winner.stopped and len(selected) < fill_to:
        needed = min(fill_to - len(selected), len(residual))
        selected.extend(node_id for node_id, _ in residual[:needed])
        residual = residual[needed:]
    selected.extend(node_id for node_id, _ in residual)
    return tuple(selected)


def prune_beam_hypotheses(
    hypotheses: list[BeamHypothesis],
    *,
    candidate_node_ids: tuple[str, ...],
    beam_size: int,
    deduplicate: bool,
) -> list[BeamHypothesis]:
    ordered = sorted(hypotheses, key=lambda item: _sort_key(item, candidate_node_ids))
    if not deduplicate:
        return ordered[:beam_size]
    result: list[BeamHypothesis] = []
    seen: set[tuple[frozenset[int], int | None]] = set()
    for hypothesis in ordered:
        if hypothesis.state_key in seen:
            continue
        seen.add(hypothesis.state_key)
        result.append(hypothesis)
        if len(result) == beam_size:
            break
    return result


def _sort_key(
    hypothesis: BeamHypothesis, candidate_node_ids: tuple[str, ...]
) -> tuple[float, float, tuple[str, ...], bool]:
    sequence = tuple(candidate_node_ids[index] for index in hypothesis.selected_indices)
    return (
        -hypothesis.normalized_score,
        -hypothesis.raw_score,
        sequence,
        not hypothesis.stopped,
    )


def normalize_beam_score(raw_score: float, action_count: int, alpha: float) -> float:
    return raw_score / (max(1, action_count) ** alpha)


def joint_action_log_probabilities(
    candidate_logits: Tensor, stop_logit: Tensor
) -> Tensor:
    if candidate_logits.ndim != 1:
        raise ValueError("candidate_logits must be one-dimensional")
    if stop_logit.numel() != 1:
        raise ValueError("stop_logit must contain exactly one value")
    action_logits = torch.cat(
        [candidate_logits, stop_logit.reshape(1)],
        dim=0,
    )
    return torch.log_softmax(action_logits, dim=0)


def _validate_config(config: BeamSearchConfig) -> None:
    if config.inference_beam_size <= 0 or config.training_beam_size <= 0:
        raise ValueError("beam size must be positive")
    if config.inference_beam_size != config.training_beam_size:
        raise ValueError("training and inference beam sizes must match")
    if not 1 <= config.max_steps <= 5:
        raise ValueError("max_steps must be in [1, 5]")


__all__ = [
    "ActionScorer",
    "BeamHypothesis",
    "BeamSearchResult",
    "complete_beam_ranking",
    "joint_action_log_probabilities",
    "normalize_beam_score",
    "prune_beam_hypotheses",
    "run_beam_search",
]
