from __future__ import annotations

from dataclasses import replace

import torch

from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.graph_retriever.batching import build_training_batches
from graph_memory.models.graph_retriever.config.records import NodeFeatureConfig
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider
from tests.test_phase2_rgcn_training import (
    FakeRetriever,
    FakeTextEmbeddingProvider,
    tiny_graphs,
    tiny_model_config,
    tiny_pairs,
    tiny_ranking_requests,
)


def _learned_model_config():
    return replace(
        tiny_model_config(),
        method_name=RetrievalMethodId.LEARNED_GRAPH_RGCN_RETRIEVER.value,
    )


def test_existing_rgcn_batches_do_not_include_learned_edge_tensors() -> None:
    batch = build_training_batches(
        ranking_requests=tiny_ranking_requests(),
        graphs=tiny_graphs(),
        pairs=tiny_pairs(),
        labels=[
            EvidenceLabel(
                task_id="hotpot_rgcn_train",
                gold_answer="Alpha",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(("m0", "m2"),),
            )
        ],
        model_config=tiny_model_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        batch_size=1,
    )[0]

    assert batch.learned_edges is None


def test_hotpotqa_empty_dependency_edges_produce_unlabeled_learned_edges() -> None:
    batch = build_training_batches(
        ranking_requests=tiny_ranking_requests(),
        graphs=tiny_graphs(),
        pairs=tiny_pairs(),
        labels=[
            EvidenceLabel(
                task_id="hotpot_rgcn_train",
                gold_answer="Alpha",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(),
            )
        ],
        model_config=_learned_model_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        batch_size=1,
    )[0]

    assert batch.learned_edges is not None
    assert batch.learned_edges.edge_features.shape == (3, 8)
    assert batch.learned_edges.edge_label_mask.tolist() == [False, False, False]
    assert batch.learned_edges.edge_labels.tolist() == [0.0, 0.0, 0.0]


def test_gold_dependency_edges_label_matching_proposal_message_edges() -> None:
    batch = build_training_batches(
        ranking_requests=tiny_ranking_requests(),
        graphs=tiny_graphs(),
        pairs=tiny_pairs(),
        labels=[
            EvidenceLabel(
                task_id="hotpot_rgcn_train",
                gold_answer="Alpha",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(("m0", "m2"),),
            )
        ],
        model_config=_learned_model_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        batch_size=1,
    )[0]

    assert batch.learned_edges is not None
    assert batch.graph_batch.edge_index.tolist() == [[0, 1, 3], [1, 3, 1]]
    assert batch.learned_edges.edge_label_mask.tolist() == [True, True, True]
    assert batch.learned_edges.edge_labels.tolist() == [0.0, 1.0, 1.0]
    torch.testing.assert_close(
        batch.learned_edges.edge_features[:, 0],
        batch.graph_batch.edge_weights,
    )


def test_learned_edge_features_use_named_node_feature_columns() -> None:
    batch = build_training_batches(
        ranking_requests=tiny_ranking_requests(),
        graphs=tiny_graphs(),
        pairs=tiny_pairs(),
        labels=[
            EvidenceLabel(
                task_id="hotpot_rgcn_train",
                gold_answer="Alpha",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(("m0", "m2"),),
            )
        ],
        model_config=replace(
            _learned_model_config(),
            feature_config=NodeFeatureConfig(
                node_feature_names=("is_question_node", "seed_rank_percentile", "seed_score"),
                scorer_feature_names=("seed_score", "seed_rank_percentile"),
            ),
        ),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        batch_size=1,
    )[0]

    assert batch.learned_edges is not None
    assert batch.graph_batch.edge_index.tolist()[0][0] == 0
    assert batch.graph_batch.edge_index.tolist()[1][0] == 1
    first_edge = batch.learned_edges.edge_features[0]
    torch.testing.assert_close(first_edge[2], torch.tensor(0.0))
    torch.testing.assert_close(first_edge[3], torch.tensor(1.0))
    torch.testing.assert_close(first_edge[4], torch.tensor(0.9))
    torch.testing.assert_close(first_edge[6], torch.tensor(1.0))
