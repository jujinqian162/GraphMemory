import torch
import torch.nn.functional as F

from graph_memory.models.graph_retriever.internals.neural import (
    EvidenceScoringModel,
    RGCNGraphEncoder,
    TypedRelationTransform,
)
from graph_memory.models.graph_retriever.config.defaults import default_model_config
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.internals.tensorization import (
    DEFAULT_RELATION_VOCAB,
)
from graph_memory.models.graph_retriever.internals.contracts import (
    GraphBatch,
    TrainingBatch,
)


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
        sample_node_features=torch.tensor(
            [[0.9, 0.0], [0.5, 0.5], [0.1, 1.0]], dtype=torch.float32
        ),
        labels=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        sample_task_ids=["hotpot_model_test", "hotpot_model_test", "hotpot_model_test"],
        sample_node_ids=["m0", "m1", "m2"],
        sample_types=["positive", "easy_random", "hard_graph_neighbor"],
    )


def test_seed_residual_starts_from_exact_seed_scores() -> None:
    config = default_model_config(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake-encoder",
        encoder_dim=3,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=64,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    model = build_model_from_config(config)
    batch = tiny_training_batch()

    assert config.scoring_mode == "seed_residual"
    assert torch.equal(model(batch), batch.sample_node_features[:, 0])


def test_wo_graph_is_exact_seed_passthrough_even_with_nonzero_head() -> None:
    config = default_model_config(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake-encoder",
        encoder_dim=3,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=64,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        ablation_name="wo_graph",
    )
    model = build_model_from_config(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    batch = tiny_training_batch()

    assert config.scoring_mode == "seed_passthrough"
    assert torch.equal(model(batch), batch.sample_node_features[:, 0])


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

    relation_params = [
        param for name, param in model.named_parameters() if "relation_linears" in name
    ]
    assert logits.shape == (3,)
    assert any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in relation_params
    )
