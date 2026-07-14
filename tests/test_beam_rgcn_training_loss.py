from __future__ import annotations

from dataclasses import replace

import torch
import pytest

from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_retriever import beam_loss as beam_loss_module
from graph_memory.models.graph_retriever.config.records import BeamSearchConfig
from graph_memory.models.graph_retriever.beam_loss import supervised_action_losses
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.training import compute_beam_training_loss
from tests.test_phase2_rgcn_model import tiny_training_batch
from tests.rgcn_fixtures import tiny_model_config, tiny_training_config


def test_beam_training_loss_combines_action_stop_and_auxiliary_terms() -> None:
    model = build_model_from_config(replace(tiny_model_config(), encoder_dim=3))
    batch = tiny_training_batch()
    label = EvidenceLabel(
        task_id="hotpot_model_test",
        gold_answer="answer",
        gold_evidence_item_ids=("m0", "m1"),
        gold_dependency_edges=(),
    )

    breakdown = compute_beam_training_loss(
        model=model,
        batch=batch,
        labels=[label],
        training_config=tiny_training_config(),
    )

    expected = (
        breakdown.next_action_loss + breakdown.stop_loss + 0.2 * breakdown.aux_node_loss
    )
    torch.testing.assert_close(breakdown.total_loss, expected)
    assert breakdown.retained_hypotheses >= 1.0
    assert 0.0 <= breakdown.oracle_reachable_rate <= 1.0
    breakdown.total_loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_joint_action_loss_calibrates_candidates_against_stop() -> None:
    baseline = supervised_action_losses(
        candidate_logits=torch.tensor([0.0, -1.0]),
        stop_logit=torch.tensor(0.0),
        eligible_candidate_indices=(0, 1),
        valid_candidate_indices=(0,),
        stop_target=False,
    )
    shifted = supervised_action_losses(
        candidate_logits=torch.tensor([5.0, 4.0]),
        stop_logit=torch.tensor(0.0),
        eligible_candidate_indices=(0, 1),
        valid_candidate_indices=(0,),
        stop_target=False,
    )

    assert not torch.isclose(baseline.next_action_loss, shifted.next_action_loss)
    assert not torch.isclose(baseline.stop_loss, shifted.stop_loss)


def test_completed_state_supervises_stop_against_distractors() -> None:
    easy = supervised_action_losses(
        candidate_logits=torch.tensor([-5.0, -4.0]),
        stop_logit=torch.tensor(0.0),
        eligible_candidate_indices=(0, 1),
        valid_candidate_indices=(),
        stop_target=True,
    )
    hard = supervised_action_losses(
        candidate_logits=torch.tensor([5.0, 4.0]),
        stop_logit=torch.tensor(0.0),
        eligible_candidate_indices=(0, 1),
        valid_candidate_indices=(),
        stop_target=True,
    )

    assert hard.next_action_loss > easy.next_action_loss
    assert hard.stop_loss > easy.stop_loss


def test_masked_future_gold_does_not_enter_supervised_denominator() -> None:
    low_future_score = supervised_action_losses(
        candidate_logits=torch.tensor([0.0, -100.0, -1.0]),
        stop_logit=torch.tensor(0.0),
        eligible_candidate_indices=(0, 2),
        valid_candidate_indices=(0,),
        stop_target=False,
    )
    high_future_score = supervised_action_losses(
        candidate_logits=torch.tensor([0.0, 100.0, -1.0]),
        stop_logit=torch.tensor(0.0),
        eligible_candidate_indices=(0, 2),
        valid_candidate_indices=(0,),
        stop_target=False,
    )

    torch.testing.assert_close(
        low_future_score.next_action_loss, high_future_score.next_action_loss
    )
    torch.testing.assert_close(low_future_score.stop_loss, high_future_score.stop_loss)


def test_training_beam_honors_disabled_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_model_from_config(replace(tiny_model_config(), encoder_dim=3))
    batch = tiny_training_batch()
    label = EvidenceLabel(
        task_id="hotpot_model_test",
        gold_answer="answer",
        gold_evidence_item_ids=("m0", "m1"),
        gold_dependency_edges=(),
    )
    observed_deduplication_values: list[bool] = []
    original_prune = beam_loss_module.prune_beam_hypotheses

    def recording_prune(*args, deduplicate: bool, **kwargs):
        observed_deduplication_values.append(deduplicate)
        return original_prune(*args, deduplicate=deduplicate, **kwargs)

    monkeypatch.setattr(beam_loss_module, "prune_beam_hypotheses", recording_prune)

    compute_beam_training_loss(
        model=model,
        batch=batch,
        labels=[label],
        training_config=tiny_training_config(),
        beam_search_config=BeamSearchConfig(deduplicate_selected_sets=False),
    )

    assert observed_deduplication_values
    assert not any(observed_deduplication_values)
