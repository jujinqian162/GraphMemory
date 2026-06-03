import pytest
import torch
import torch.nn.functional as F

from graph_memory.models.graph_retriever.internals.neural import (
    EvidenceNodeScorer,
    EvidenceScoringModel,
    IdentityGraphEncoder,
    RGCNGraphEncoder,
    SharedRelationTransform,
    TypedRelationTransform,
)
from graph_memory.models.graph_retriever.internals.tensorization import DEFAULT_RELATION_VOCAB
from graph_memory.models.graph_retriever.internals.contracts import GraphBatch, TrainingBatch


def tiny_graph_batch() -> GraphBatch:
    return GraphBatch(
        node_embeddings=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        node_features=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.9, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.1, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 1]], dtype=torch.long),
        relation_ids=torch.tensor([0, 5, 6], dtype=torch.long),
        edge_weights=torch.tensor([1.0, 0.7, 0.7], dtype=torch.float32),
        query_node_indices=torch.tensor([0], dtype=torch.long),
        task_node_offsets=[0, 4],
        task_ids=["hotpot_model_test"],
        node_ids_by_task=[["q", "m0", "m1", "m2"]],
    )


def tiny_training_batch() -> TrainingBatch:
    graph_batch = tiny_graph_batch()
    return TrainingBatch(
        graph_batch=graph_batch,
        sample_node_indices=torch.tensor([1, 2, 3], dtype=torch.long),
        sample_query_indices=torch.tensor([0, 0, 0], dtype=torch.long),
        sample_node_features=torch.tensor([[0.9, 0.0], [0.5, 0.5], [0.1, 1.0]], dtype=torch.float32),
        labels=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        sample_task_ids=["hotpot_model_test", "hotpot_model_test", "hotpot_model_test"],
        sample_node_ids=["m0", "m1", "m2"],
        sample_types=["positive", "easy_random", "hard_graph_neighbor"],
    )


def test_identity_graph_encoder_returns_input_states():
    batch = tiny_graph_batch()
    node_states = torch.randn(4, 8)

    output = IdentityGraphEncoder().forward(batch, node_states)

    assert output is node_states


def test_identity_graph_encoder_supports_torch_module_device_transfer():
    model = EvidenceScoringModel(
        encoder_dim=3,
        node_feature_dim=3,
        hidden_dim=8,
        graph_encoder=IdentityGraphEncoder(),
        scorer_feature_dim=2,
        dropout=0.0,
    )

    assert model.to("cpu").graph_encoder is model.graph_encoder


def test_rgcn_graph_encoder_supports_typed_and_shared_relation_transforms():
    batch = tiny_graph_batch()
    node_states = torch.randn(4, 8)

    typed_encoder = RGCNGraphEncoder(
        hidden_dim=8,
        num_relations=len(DEFAULT_RELATION_VOCAB),
        num_layers=1,
        message_transform_factory=lambda: TypedRelationTransform(
            hidden_dim=8,
            num_relations=len(DEFAULT_RELATION_VOCAB),
        ),
        dropout=0.0,
    )
    shared_encoder = RGCNGraphEncoder(
        hidden_dim=8,
        num_relations=len(DEFAULT_RELATION_VOCAB),
        num_layers=1,
        message_transform_factory=lambda: SharedRelationTransform(hidden_dim=8),
        dropout=0.0,
    )

    assert typed_encoder.forward(batch, node_states).shape == node_states.shape
    assert shared_encoder.forward(batch, node_states).shape == node_states.shape


def test_rgcn_graph_encoder_applies_shared_transform_to_every_layer():
    encoder = RGCNGraphEncoder(
        hidden_dim=8,
        num_relations=len(DEFAULT_RELATION_VOCAB),
        num_layers=2,
        message_transform_factory=lambda: SharedRelationTransform(hidden_dim=8),
        dropout=0.0,
    )

    assert [type(layer.message_transform) for layer in encoder.layers] == [
        SharedRelationTransform,
        SharedRelationTransform,
    ]


def test_rgcn_graph_encoder_builds_independent_typed_transform_per_layer():
    encoder = RGCNGraphEncoder(
        hidden_dim=8,
        num_relations=len(DEFAULT_RELATION_VOCAB),
        num_layers=2,
        message_transform_factory=lambda: TypedRelationTransform(
            hidden_dim=8,
            num_relations=len(DEFAULT_RELATION_VOCAB),
        ),
        dropout=0.0,
    )

    transforms = [layer.message_transform for layer in encoder.layers]
    assert all(isinstance(transform, TypedRelationTransform) for transform in transforms)
    assert transforms[0] is not transforms[1]


def test_typed_relation_transform_rejects_invalid_relation_ids():
    transform = TypedRelationTransform(hidden_dim=8, num_relations=2)

    with pytest.raises(ValueError, match="relation_ids"):
        transform.forward(torch.randn(2, 8), torch.tensor([0, 2], dtype=torch.long))

    with pytest.raises(ValueError, match="same length"):
        transform.forward(torch.randn(2, 8), torch.tensor([0], dtype=torch.long))


def test_evidence_node_scorer_returns_one_logit_per_sample():
    scorer = EvidenceNodeScorer(hidden_dim=8, scorer_feature_dim=2, dropout=0.0)

    logits = scorer(
        node_states=torch.randn(3, 8),
        query_states=torch.randn(3, 8),
        sample_node_features=torch.randn(3, 2),
    )

    assert logits.shape == (3,)


def test_evidence_scoring_model_backward_updates_relation_parameters():
    torch.manual_seed(13)
    batch = tiny_training_batch()
    graph_encoder = RGCNGraphEncoder(
        hidden_dim=8,
        num_relations=len(DEFAULT_RELATION_VOCAB),
        num_layers=1,
        message_transform_factory=lambda: TypedRelationTransform(
            hidden_dim=8,
            num_relations=len(DEFAULT_RELATION_VOCAB),
        ),
        dropout=0.0,
    )
    model = EvidenceScoringModel(
        encoder_dim=3,
        node_feature_dim=3,
        hidden_dim=8,
        graph_encoder=graph_encoder,
        scorer_feature_dim=2,
        dropout=0.0,
    )

    logits = model(batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
    loss.backward()

    relation_params = [param for name, param in model.named_parameters() if "relation_linears" in name]
    assert logits.shape == (3,)
    assert any(param.grad is not None and torch.any(param.grad != 0) for param in relation_params)
