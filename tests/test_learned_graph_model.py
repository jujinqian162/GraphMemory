from __future__ import annotations

from dataclasses import replace

import torch

from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.internals.neural import (
    EdgeGateScorer,
    EvidenceScoringOutput,
    GatedRGCNGraphEncoder,
    LearnedGraphEvidenceScoringModel,
    RGCNGraphEncoder,
    TypedRelationTransform,
)
from graph_memory.models.graph_retriever.internals.tensorization import DEFAULT_RELATION_VOCAB
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider
from tests.test_learned_graph_edge_labels import _learned_model_config
from tests.test_phase2_rgcn_model import tiny_graph_batch
from tests.test_phase2_rgcn_training import (
    FakeRetriever,
    FakeTextEmbeddingProvider,
    tiny_graphs,
    tiny_pairs,
    tiny_ranking_requests,
)
from graph_memory.models.graph_retriever.batching import build_training_batches


def _learned_batch():
    return build_training_batches(
        ranking_requests=tiny_ranking_requests(),
        graphs=tiny_graphs(),
        pairs=tiny_pairs(),
        model_config=_learned_model_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        batch_size=1,
    )[0]


def test_edge_gate_scorer_returns_one_logit_per_message_edge() -> None:
    scorer = EdgeGateScorer(feature_dim=8, hidden_dim=4, dropout=0.0)

    logits = scorer(torch.randn(3, 8))

    assert logits.shape == (3,)


def test_factory_builds_learned_graph_model_with_edge_outputs() -> None:
    batch = _learned_batch()
    model = build_model_from_config(_learned_model_config())

    output = model(batch)

    assert isinstance(model, LearnedGraphEvidenceScoringModel)
    assert isinstance(output, EvidenceScoringOutput)
    assert output.node_logits.shape == batch.labels.shape
    assert output.edge_gate_logits is not None
    assert output.edge_gates is not None
    assert output.edge_gate_logits.shape == batch.graph_batch.edge_weights.shape
    assert output.edge_gates.shape == batch.graph_batch.edge_weights.shape


def test_zero_edge_gates_match_zero_weight_message_passing() -> None:
    torch.manual_seed(13)
    graph_batch = tiny_graph_batch()
    node_states = torch.randn(4, 8)
    base_encoder = RGCNGraphEncoder(
        hidden_dim=8,
        num_relations=len(DEFAULT_RELATION_VOCAB),
        num_layers=1,
        message_transform_factory=lambda: TypedRelationTransform(
            hidden_dim=8,
            num_relations=len(DEFAULT_RELATION_VOCAB),
        ),
        dropout=0.0,
    )
    encoder = GatedRGCNGraphEncoder(base_encoder=base_encoder)
    zero_graph_batch = replace(
        graph_batch,
        edge_weights=torch.zeros_like(graph_batch.edge_weights),
    )

    actual = encoder.forward(graph_batch, node_states, edge_gates=torch.zeros_like(graph_batch.edge_weights))
    expected = base_encoder.forward(zero_graph_batch, node_states)

    torch.testing.assert_close(actual, expected)


def test_existing_rgcn_model_still_returns_tensor() -> None:
    config = replace(
        _learned_model_config(),
        method_name=RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER.value,
    )
    batch = _learned_batch()
    model = build_model_from_config(config)

    output = model(batch)

    assert isinstance(output, torch.Tensor)
