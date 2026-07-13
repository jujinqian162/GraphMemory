from __future__ import annotations

import torch
import pytest

from graph_memory.models.graph_retriever.config.records import BeamSearchConfig
from graph_memory.models.graph_retriever.decoder import (
    BeamHypothesis,
    complete_beam_ranking,
    normalize_beam_score,
    prune_beam_hypotheses,
    run_beam_search,
)


def test_beam_search_is_bounded_unique_and_deterministic() -> None:
    node_ids = ("a", "b", "c")

    def score(selected: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.tensor([1.0, 1.0, 0.0])
        stop = torch.tensor(-10.0 if len(selected) < 2 else 10.0)
        return logits, stop

    result = run_beam_search(
        candidate_node_ids=node_ids,
        score_actions=score,
        config=BeamSearchConfig(max_steps=5),
    )

    assert result.winner.selected_indices == (0, 1)
    assert result.winner.stopped is True
    assert len(set(result.winner.selected_indices)) == len(
        result.winner.selected_indices
    )
    assert result.steps <= 5


def test_state_deduplication_uses_selected_set_and_last_node() -> None:
    higher_same_state = BeamHypothesis(
        selected_indices=(0, 1, 2), raw_score=-0.2, normalized_score=-0.2
    )
    lower_same_state = BeamHypothesis(
        selected_indices=(1, 0, 2), raw_score=-0.4, normalized_score=-0.4
    )
    different_last_node = BeamHypothesis(
        selected_indices=(0, 2, 1), raw_score=-0.3, normalized_score=-0.3
    )

    pruned = prune_beam_hypotheses(
        [lower_same_state, different_last_node, higher_same_state],
        candidate_node_ids=("a", "b", "c"),
        beam_size=3,
        deduplicate=True,
    )

    assert [item.selected_indices for item in pruned] == [(0, 1, 2), (0, 2, 1)]


def test_state_deduplication_can_be_disabled() -> None:
    hypotheses = [
        BeamHypothesis(
            selected_indices=(0, 1, 2), raw_score=-0.2, normalized_score=-0.2
        ),
        BeamHypothesis(
            selected_indices=(1, 0, 2), raw_score=-0.4, normalized_score=-0.4
        ),
    ]

    pruned = prune_beam_hypotheses(
        hypotheses,
        candidate_node_ids=("a", "b", "c"),
        beam_size=2,
        deduplicate=False,
    )

    assert len(pruned) == 2


def test_beam_action_count_and_normalization_include_stop() -> None:
    active = BeamHypothesis(selected_indices=(0, 1), raw_score=-3.0)
    stopped = BeamHypothesis(selected_indices=(0, 1), raw_score=-3.0, stopped=True)
    empty_stop = BeamHypothesis(raw_score=-3.0, stopped=True)

    assert active.action_count == 2
    assert stopped.action_count == 3
    assert empty_stop.action_count == 1
    assert normalize_beam_score(active.raw_score, active.action_count, 1.0) == -1.5
    assert normalize_beam_score(stopped.raw_score, stopped.action_count, 1.0) == -1.0
    assert (
        normalize_beam_score(empty_stop.raw_score, empty_stop.action_count, 1.0) == -3.0
    )


def test_early_stop_ranking_keeps_prefix_then_fills_by_base_score() -> None:
    ranking = complete_beam_ranking(
        candidate_node_ids=("a", "b", "c", "d"),
        base_logits=torch.tensor([0.2, 0.9, 0.7, 0.1]),
        winner=BeamHypothesis(
            selected_indices=(0,),
            raw_score=-0.1,
            stopped=True,
            stop_score=2.0,
        ),
        fill_to=5,
    )

    assert ranking == ("a", "b", "c", "d")


@pytest.mark.parametrize("beam_size", [1, 2, 4])
def test_beam_search_supports_bounded_comparison_widths(beam_size: int) -> None:
    result = run_beam_search(
        candidate_node_ids=("a", "b", "c", "d"),
        score_actions=lambda selected: (
            torch.arange(4, dtype=torch.float32),
            torch.tensor(-2.0),
        ),
        config=BeamSearchConfig(
            training_beam_size=beam_size,
            inference_beam_size=beam_size,
            max_steps=2,
        ),
    )
    assert len(result.final_beam) <= beam_size
