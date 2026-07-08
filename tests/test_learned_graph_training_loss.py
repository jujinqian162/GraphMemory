from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F

from graph_memory.models.graph_retriever.config.records import RgcnLossConfig
from graph_memory.models.graph_retriever.config.records import RgcnTrainingConfig
from graph_memory.models.graph_retriever.internals.contracts import LearnedEdgeBatch
from graph_memory.models.graph_retriever.internals.neural import EvidenceScoringOutput
from graph_memory.models.graph_retriever.training import _compute_loss_components, train_graph_retriever
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider
from tests.test_learned_graph_edge_labels import _learned_model_config
from tests.test_phase2_rgcn_training import (
    FakeRetriever,
    FakeTextEmbeddingProvider,
    evidence_labels,
    tiny_graphs,
    tiny_labels,
    tiny_pairs,
    tiny_ranking_requests,
)


def _learned_edges(mask: list[bool], labels: list[float]) -> LearnedEdgeBatch:
    return LearnedEdgeBatch(
        edge_features=torch.empty((len(mask), 8), dtype=torch.float32),
        edge_label_mask=torch.tensor(mask, dtype=torch.bool),
        edge_labels=torch.tensor(labels, dtype=torch.float32),
    )


def test_rgcn_loss_config_defaults_are_fixed_design_values() -> None:
    config = RgcnLossConfig()

    assert config.rank_loss_weight == 1.0
    assert config.edge_loss_weight == 0.2
    assert config.sparse_loss_weight == 0.05
    assert config.pairwise_rank_loss_weight == 0.0
    assert config.pairwise_temperature == 1.0
    assert config.pairwise_negative_type_weights == {
        "easy_random": 1.0,
        "hard_bm25": 1.0,
        "hard_dense": 1.0,
        "hard_graph_neighbor": 1.0,
    }


def test_learned_graph_total_loss_is_weighted_rank_edge_sparse_sum() -> None:
    loss_config = RgcnLossConfig(rank_loss_weight=1.0, edge_loss_weight=0.2, sparse_loss_weight=0.05)
    node_logits = torch.tensor([0.25, -0.5], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    edge_gate_logits = torch.tensor([0.75, -0.25], dtype=torch.float32)
    edge_gates = torch.sigmoid(edge_gate_logits)
    learned_edges = _learned_edges([True, True], [1.0, 0.0])

    components = _compute_loss_components(
        output=EvidenceScoringOutput(
            node_logits=node_logits,
            edge_gate_logits=edge_gate_logits,
            edge_gates=edge_gates,
        ),
        labels=labels,
        sample_task_ids=["task-1", "task-1"],
        sample_types=["positive", "easy_random"],
        learned_edges=learned_edges,
        loss_config=loss_config,
        pos_weight=None,
    )

    expected_rank = F.binary_cross_entropy_with_logits(node_logits, labels)
    expected_edge = F.binary_cross_entropy_with_logits(edge_gate_logits, learned_edges.edge_labels)
    expected_sparse = edge_gates.mean()
    expected_total = expected_rank + 0.2 * expected_edge + 0.05 * expected_sparse
    torch.testing.assert_close(components.total_loss, expected_total)
    torch.testing.assert_close(components.rank_loss, expected_rank)
    torch.testing.assert_close(components.edge_loss, expected_edge)
    torch.testing.assert_close(components.sparse_loss, expected_sparse)
    assert components.edge_loss_sample_count == 2
    assert math.isclose(components.mean_edge_gate, float(expected_sparse))


def test_learned_graph_edge_loss_is_zero_without_labeled_edges() -> None:
    components = _compute_loss_components(
        output=EvidenceScoringOutput(
            node_logits=torch.tensor([0.0], dtype=torch.float32),
            edge_gate_logits=torch.tensor([10.0, -10.0], dtype=torch.float32),
            edge_gates=torch.tensor([1.0, 0.0], dtype=torch.float32),
        ),
        labels=torch.tensor([1.0], dtype=torch.float32),
        sample_task_ids=["task-1"],
        sample_types=["positive"],
        learned_edges=_learned_edges([False, False], [0.0, 0.0]),
        loss_config=RgcnLossConfig(),
        pos_weight=None,
    )

    assert components.edge_loss.item() == 0.0
    assert components.edge_loss_sample_count == 0


def test_zero_edge_loss_weight_removes_edge_bce_from_total_loss() -> None:
    node_logits = torch.tensor([0.0], dtype=torch.float32)
    labels = torch.tensor([1.0], dtype=torch.float32)
    edge_gate_logits = torch.tensor([-100.0], dtype=torch.float32)
    edge_gates = torch.sigmoid(edge_gate_logits)
    components = _compute_loss_components(
        output=EvidenceScoringOutput(
            node_logits=node_logits,
            edge_gate_logits=edge_gate_logits,
            edge_gates=edge_gates,
        ),
        labels=labels,
        sample_task_ids=["task-1"],
        sample_types=["positive"],
        learned_edges=_learned_edges([True], [1.0]),
        loss_config=RgcnLossConfig(rank_loss_weight=1.0, edge_loss_weight=0.0, sparse_loss_weight=0.0),
        pos_weight=None,
    )

    expected_rank = F.binary_cross_entropy_with_logits(node_logits, labels)
    torch.testing.assert_close(components.total_loss, expected_rank)
    assert components.edge_loss.item() > 0.0


def test_pairwise_rank_loss_penalizes_same_task_negatives_by_type_weight() -> None:
    node_logits = torch.tensor([0.25, 1.25], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss_config = RgcnLossConfig(
        rank_loss_weight=0.0,
        edge_loss_weight=0.0,
        sparse_loss_weight=0.0,
        pairwise_rank_loss_weight=1.0,
        pairwise_temperature=0.5,
        pairwise_negative_type_weights={
            "easy_random": 0.5,
            "hard_bm25": 1.2,
            "hard_dense": 1.5,
            "hard_graph_neighbor": 1.3,
        },
    )

    components = _compute_loss_components(
        output=node_logits,
        labels=labels,
        sample_task_ids=["task-1", "task-1"],
        sample_types=["positive", "hard_dense"],
        learned_edges=None,
        loss_config=loss_config,
        pos_weight=None,
    )

    expected_pairwise = F.softplus(-((node_logits[0] - node_logits[1]) / 0.5))
    torch.testing.assert_close(components.pairwise_rank_loss, expected_pairwise)
    torch.testing.assert_close(components.total_loss, expected_pairwise)
    assert components.pairwise_pair_count == 1


def test_pairwise_rank_loss_ignores_cross_task_samples() -> None:
    components = _compute_loss_components(
        output=torch.tensor([0.0, 1.0], dtype=torch.float32),
        labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        sample_task_ids=["task-1", "task-2"],
        sample_types=["positive", "hard_dense"],
        learned_edges=None,
        loss_config=RgcnLossConfig(
            rank_loss_weight=0.0,
            edge_loss_weight=0.0,
            sparse_loss_weight=0.0,
            pairwise_rank_loss_weight=1.0,
        ),
        pos_weight=None,
    )

    assert components.pairwise_rank_loss.item() == 0.0
    assert components.pairwise_pair_count == 0
    assert components.total_loss.item() == 0.0


def test_pairwise_rank_loss_uses_weighted_mean_over_valid_pairs() -> None:
    node_logits = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    loss_config = RgcnLossConfig(
        rank_loss_weight=0.0,
        edge_loss_weight=0.0,
        sparse_loss_weight=0.0,
        pairwise_rank_loss_weight=1.0,
        pairwise_negative_type_weights={
            "easy_random": 0.5,
            "hard_bm25": 1.2,
            "hard_dense": 1.5,
            "hard_graph_neighbor": 1.3,
        },
    )

    components = _compute_loss_components(
        output=node_logits,
        labels=labels,
        sample_task_ids=["task-1", "task-1", "task-1"],
        sample_types=["positive", "easy_random", "hard_graph_neighbor"],
        learned_edges=None,
        loss_config=loss_config,
        pos_weight=None,
    )

    easy_loss = F.softplus(-(node_logits[0] - node_logits[1]))
    graph_loss = F.softplus(-(node_logits[0] - node_logits[2]))
    expected_pairwise = (0.5 * easy_loss + 1.3 * graph_loss) / (0.5 + 1.3)
    torch.testing.assert_close(components.pairwise_rank_loss, expected_pairwise)
    assert components.pairwise_pair_count == 2


def test_learned_graph_training_metrics_record_loss_components_and_weights() -> None:
    result = train_graph_retriever(
        train_requests=tiny_ranking_requests(),
        train_graphs=tiny_graphs(),
        train_pairs=tiny_pairs(),
        train_labels=[
            EvidenceLabel(
                task_id="hotpot_rgcn_train",
                gold_answer="Alpha",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(("m0", "m2"),),
            )
        ],
        dev_requests=tiny_ranking_requests(),
        dev_labels=evidence_labels(tiny_labels()),
        dev_graphs=tiny_graphs(),
        model_config=_learned_model_config(),
        training_config=RgcnTrainingConfig(
            optimizer_name="AdamW",
            learning_rate=0.01,
            batch_size=1,
            max_grad_norm=1.0,
            random_seed=13,
            pos_weight_enabled=False,
            epochs=1,
            loss_config=RgcnLossConfig(
                rank_loss_weight=1.0,
                edge_loss_weight=0.2,
                sparse_loss_weight=0.05,
                pairwise_rank_loss_weight=1.0,
            ),
        ),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
    )

    record = result.metric_records[0]
    assert cast(float, record["train_rank_loss"]) > 0.0
    assert cast(float, record["train_edge_loss"]) > 0.0
    assert cast(float, record["train_sparse_loss"]) > 0.0
    assert record["edge_loss_sample_count"] == 3
    assert 0.0 <= cast(float, record["mean_edge_gate"]) <= 1.0
    assert record["rank_loss_weight"] == 1.0
    assert record["edge_loss_weight"] == 0.2
    assert record["sparse_loss_weight"] == 0.05
    assert cast(float, record["train_pairwise_rank_loss"]) > 0.0
    assert record["pairwise_pair_count"] == 2
    assert record["pairwise_rank_loss_weight"] == 1.0
    assert record["pairwise_temperature"] == 1.0
    assert record["pairwise_negative_type_weights"] == {
        "easy_random": 1.0,
        "hard_bm25": 1.0,
        "hard_dense": 1.0,
        "hard_graph_neighbor": 1.0,
    }
