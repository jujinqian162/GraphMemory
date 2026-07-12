from __future__ import annotations

import torch
import pytest

from graph_memory.models.graph_retriever.config.records import BeamSearchConfig
from graph_memory.models.graph_retriever.decoder import (
    BeamHypothesis,
    complete_beam_ranking,
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


def test_set_deduplication_retains_one_higher_scoring_order() -> None:
    left = BeamHypothesis(selected_indices=(0, 1), raw_score=-0.2)
    right = BeamHypothesis(selected_indices=(1, 0), raw_score=-0.4)

    result = run_beam_search(
        candidate_node_ids=("a", "b"),
        score_actions=lambda selected: (torch.zeros(2), torch.tensor(10.0)),
        config=BeamSearchConfig(
            training_beam_size=2, inference_beam_size=2, max_steps=1
        ),
        initial_hypotheses=(left, right),
    )

    matching = [
        hypothesis
        for hypothesis in result.final_beam
        if hypothesis.selected_set == frozenset({0, 1})
    ]
    assert len(matching) == 1
    assert matching[0].selected_indices == (0, 1)


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
