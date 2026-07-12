from __future__ import annotations

from dataclasses import replace

import torch

from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.training import compute_beam_training_loss
from tests.test_phase2_rgcn_model import tiny_training_batch
from tests.test_phase2_rgcn_training import tiny_model_config, tiny_training_config


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
