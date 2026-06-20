from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from graph_memory.contracts.graphs import GraphEdge, GraphItemNode, MemoryGraph
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTextRankingRequest
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord, HotpotQALabelRecord
from graph_memory.contracts.training_pairs import TrainPairRecord
from graph_memory.models.graph_retriever.batching import (
    build_full_ranking_batches,
    build_training_batches,
    move_training_batch,
)
from graph_memory.models.graph_retriever.config.records import (
    NodeFeatureConfig,
    RgcnModelConfig,
    RgcnTrainingConfig,
)
from graph_memory.models.graph_retriever.text_embeddings import DenseGraphFeatureProvider
from graph_memory.models.graph_retriever.training import train_graph_retriever
from graph_memory.retrieval.methods.flat.dense import DenseConfig, DenseTaskRetriever
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider, SeedSignal


class RecordingEncoder:
    def __init__(self, vectors_by_text: Mapping[str, Sequence[float]]) -> None:
        self.vectors_by_text = vectors_by_text
        self.calls: list[tuple[list[str], int, bool]] = []

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> object:
        text_list = list(texts)
        self.calls.append((text_list, batch_size, normalize_embeddings))
        return np.asarray([self.vectors_by_text[text] for text in text_list], dtype=float)

    def get_sentence_embedding_dimension(self) -> int:
        return 4


def _task(task_id: str, query: str, node_ids: list[str]) -> HotpotQARankingRecord:
    return {
        "task_id": task_id,
        "question": query,
        "candidate_sentences": [
            {
                "sentence_id": node_id,
                "text": f"text-{node_id}",
                "title": f"source-{node_id}",
                "sentence_index": index,
                "position": index,
            }
            for index, node_id in enumerate(node_ids)
        ],
    }


def _ranking_requests(tasks: Sequence[HotpotQARankingRecord]) -> list[TextRankingRequest]:
    projector = HotpotQAToTextRankingRequest()
    return [projector.project(task) for task in tasks]


def _evidence_labels(labels: Sequence[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
        )
        for label in labels
    ]


def _graph_nodes(task_input: HotpotQARankingRecord) -> list[GraphItemNode]:
    return [
        {
            "id": sentence["sentence_id"],
            "node_type": "graph_item",
            "node_kind": "document_sentence",
            "text": sentence["text"],
            "source_ref": sentence["title"],
            "group_key": f"document:{sentence['title']}",
            "sequence_index": sentence["sentence_index"],
            "metadata": {"title": sentence["title"], "position": sentence["position"]},
        }
        for sentence in task_input["candidate_sentences"]
    ]


def _graph(task_input: HotpotQARankingRecord) -> MemoryGraph:
    memory_ids = [item["sentence_id"] for item in task_input["candidate_sentences"]]
    edges: list[GraphEdge] = [
        {
            "source": "q",
            "target": memory_ids[0],
            "edge_type": "query_overlap",
            "weight": 1.0,
            "directed": True,
        }
    ]
    if len(memory_ids) > 1:
        edges.append(
            {
                "source": memory_ids[0],
                "target": memory_ids[1],
                "edge_type": "bridge",
                "weight": 0.5,
                "directed": False,
            }
        )
    return {
        "task_id": task_input["task_id"],
        "nodes": [{"id": "q", "node_type": "question", "text": task_input["question"]}, *_graph_nodes(task_input)],
        "edges": edges,
    }


def _model_config() -> RgcnModelConfig:
    return RgcnModelConfig(
        method_name="dense_rgcn_graph_retriever",
        encoder_model="fake",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=64,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        feature_config=NodeFeatureConfig(),
        relation_vocab=(
            "query_overlap_forward",
            "sequential_forward",
            "sequential_reverse",
            "entity_overlap_forward",
            "entity_overlap_reverse",
            "bridge_forward",
            "bridge_reverse",
        ),
        graph_encoder_type="rgcn",
        message_transform_type="typed",
        edge_weight_policy="artifact",
        enabled_edge_types=("bridge", "entity_overlap", "query_overlap", "sequential"),
        ablation_name="full_rgcn",
    )


def _fixture() -> tuple[
    list[HotpotQARankingRecord],
    list[MemoryGraph],
    list[TrainPairRecord],
    Mapping[str, Sequence[float]],
]:
    tasks = [
        _task("t1", "first", ["m0", "m1"]),
        _task("t2", "second", ["m0"]),
    ]
    pairs: list[TrainPairRecord] = [
        {"task_id": "t1", "node_id": "m0", "label": 1, "sample_type": "positive"},
        {"task_id": "t1", "node_id": "m1", "label": 0, "sample_type": "easy_random"},
        {"task_id": "t2", "node_id": "m0", "label": 1, "sample_type": "positive"},
    ]
    vectors = {
        "query: first": [1.0, 0.0, 0.0, 0.0],
        "passage: source-m0. text-m0": [0.8, 0.1, 0.0, 0.0],
        "passage: source-m1. text-m1": [0.2, 0.9, 0.0, 0.0],
        "query: second": [0.0, 1.0, 0.0, 0.0],
    }
    return tasks, [_graph(task) for task in tasks], pairs, vectors


def _assert_batches_equal(actual, expected) -> None:
    actual_graph = actual.graph_batch
    expected_graph = expected.graph_batch
    torch.testing.assert_close(actual_graph.node_embeddings, expected_graph.node_embeddings)
    torch.testing.assert_close(actual_graph.node_features, expected_graph.node_features)
    torch.testing.assert_close(actual_graph.edge_index, expected_graph.edge_index)
    torch.testing.assert_close(actual_graph.relation_ids, expected_graph.relation_ids)
    torch.testing.assert_close(actual_graph.edge_weights, expected_graph.edge_weights)
    torch.testing.assert_close(actual_graph.query_node_indices, expected_graph.query_node_indices)
    assert actual_graph.node_embeddings.dtype == expected_graph.node_embeddings.dtype == torch.float32
    assert actual_graph.node_features.dtype == expected_graph.node_features.dtype == torch.float32
    assert actual_graph.edge_index.dtype == expected_graph.edge_index.dtype == torch.long
    assert actual_graph.relation_ids.dtype == expected_graph.relation_ids.dtype == torch.long
    assert actual_graph.edge_weights.dtype == expected_graph.edge_weights.dtype == torch.float32
    assert actual_graph.task_node_offsets == expected_graph.task_node_offsets
    assert actual_graph.task_ids == expected_graph.task_ids
    assert actual_graph.node_ids_by_task == expected_graph.node_ids_by_task
    torch.testing.assert_close(actual.sample_node_indices, expected.sample_node_indices)
    torch.testing.assert_close(actual.sample_query_indices, expected.sample_query_indices)
    torch.testing.assert_close(actual.sample_node_features, expected.sample_node_features)
    torch.testing.assert_close(actual.labels, expected.labels)
    assert actual.sample_task_ids == expected.sample_task_ids
    assert actual.sample_node_ids == expected.sample_node_ids
    assert actual.sample_types == expected.sample_types


def test_joint_dense_graph_features_match_separate_providers_with_one_encoder_call() -> None:
    tasks, graphs, pairs, vectors = _fixture()
    joint_encoder = RecordingEncoder(vectors)
    joint_provider = DenseGraphFeatureProvider(encoder=joint_encoder, batch_size=5)
    joint_batch = build_training_batches(
        ranking_requests=_ranking_requests(tasks),
        graphs=graphs,
        pairs=pairs,
        model_config=_model_config(),
        text_embedding_provider=joint_provider,
        seed_signal_provider=joint_provider,
        batch_size=2,
    )[0]

    text_encoder = RecordingEncoder(vectors)
    seed_encoder = RecordingEncoder(vectors)
    text_provider = DenseGraphFeatureProvider(encoder=text_encoder, batch_size=5)
    seed_provider = RetrieverSeedSignalProvider(
        DenseTaskRetriever(config=DenseConfig(batch_size=5), encoder=seed_encoder)
    )
    separate_batch = build_training_batches(
        ranking_requests=_ranking_requests(tasks),
        graphs=graphs,
        pairs=pairs,
        model_config=_model_config(),
        text_embedding_provider=text_provider,
        seed_signal_provider=seed_provider,
        batch_size=2,
    )[0]

    _assert_batches_equal(joint_batch, separate_batch)
    assert len(joint_encoder.calls) == 1
    assert len(text_encoder.calls) == 1
    assert len(seed_encoder.calls) == 1
    assert joint_encoder.calls[0][0] == [
        "query: first",
        "passage: source-m0. text-m0",
        "passage: source-m1. text-m1",
        "query: second",
        "passage: source-m0. text-m0",
    ]


def test_independently_injected_seed_provider_keeps_its_semantics() -> None:
    tasks, graphs, pairs, vectors = _fixture()
    text_provider = DenseGraphFeatureProvider(encoder=RecordingEncoder(vectors))

    class CustomSeedProvider:
        def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
            return [
                SeedSignal(
                    node_id=item.item_id,
                    score=9.0 - index,
                    rank=index + 1,
                    rank_percentile=float(index),
                )
                for index, item in enumerate(reversed(request.candidates))
            ]

    batch = build_training_batches(
        ranking_requests=_ranking_requests(tasks),
        graphs=graphs,
        pairs=pairs,
        model_config=_model_config(),
        text_embedding_provider=text_provider,
        seed_signal_provider=CustomSeedProvider(),
        batch_size=2,
    )[0]

    first_task_features = batch.graph_batch.node_features[:3]
    assert first_task_features[:, 0].tolist() == [0.0, 8.0, 9.0]


def test_multi_epoch_training_builds_frozen_dev_features_once_but_evaluates_each_epoch() -> None:
    tasks, graphs, pairs, vectors = _fixture()
    labels: list[HotpotQALabelRecord] = [
        {
            "task_id": "t1",
            "gold_answer": "answer",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        },
        {
            "task_id": "t2",
            "gold_answer": "answer",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        },
    ]
    encoder = RecordingEncoder(vectors)
    provider = DenseGraphFeatureProvider(encoder=encoder, batch_size=8)

    result = train_graph_retriever(
        train_requests=_ranking_requests(tasks),
        train_graphs=graphs,
        train_pairs=pairs,
        train_labels=_evidence_labels(labels),
        dev_requests=_ranking_requests(tasks),
        dev_labels=_evidence_labels(labels),
        dev_graphs=graphs,
        model_config=_model_config(),
        training_config=RgcnTrainingConfig(
            optimizer_name="AdamW",
            learning_rate=0.01,
            batch_size=2,
            max_grad_norm=1.0,
            random_seed=13,
            pos_weight_enabled=False,
            epochs=3,
        ),
        text_embedding_provider=provider,
        seed_signal_provider=provider,
    )

    assert len(encoder.calls) == 2
    assert len(result.metric_records) == 3
    assert [record["epoch"] for record in result.metric_records] == [1, 2, 3]


def test_reused_cpu_batches_are_not_mutated_by_device_movement() -> None:
    tasks, graphs, _, vectors = _fixture()
    labels: list[HotpotQALabelRecord] = [
        {
            "task_id": task["task_id"],
            "gold_answer": "answer",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        }
        for task in tasks
    ]
    provider = DenseGraphFeatureProvider(encoder=RecordingEncoder(vectors))
    batch = build_full_ranking_batches(
        ranking_requests=_ranking_requests(tasks),
        graphs=graphs,
        labels=_evidence_labels(labels),
        model_config=_model_config(),
        text_embedding_provider=provider,
        seed_signal_provider=provider,
        batch_size=2,
    )[0]
    original_embeddings = batch.graph_batch.node_embeddings.clone()
    original_features = batch.graph_batch.node_features.clone()

    first_move = move_training_batch(batch, "cpu")
    second_move = move_training_batch(batch, "cpu")

    assert first_move is not batch
    assert second_move is not batch
    torch.testing.assert_close(batch.graph_batch.node_embeddings, original_embeddings)
    torch.testing.assert_close(batch.graph_batch.node_features, original_features)
    assert batch.graph_batch.node_embeddings.device.type == "cpu"


def test_frozen_feature_reuse_does_not_cross_training_invocations() -> None:
    tasks, graphs, pairs, vectors = _fixture()
    labels: list[HotpotQALabelRecord] = [
        {
            "task_id": task["task_id"],
            "gold_answer": "answer",
            "gold_evidence_sentence_ids": ["m0"],
            "gold_dependency_edges": [],
        }
        for task in tasks
    ]
    encoder = RecordingEncoder(vectors)
    provider = DenseGraphFeatureProvider(encoder=encoder)
    training_config = RgcnTrainingConfig(
        optimizer_name="AdamW",
        learning_rate=0.01,
        batch_size=2,
        max_grad_norm=1.0,
        random_seed=13,
        pos_weight_enabled=False,
        epochs=1,
    )

    for _ in range(2):
        train_graph_retriever(
            train_requests=_ranking_requests(tasks),
            train_graphs=graphs,
            train_pairs=pairs,
            train_labels=_evidence_labels(labels),
            dev_requests=_ranking_requests(tasks),
            dev_labels=_evidence_labels(labels),
            dev_graphs=graphs,
            model_config=_model_config(),
            training_config=training_config,
            text_embedding_provider=provider,
            seed_signal_provider=provider,
        )

    assert len(encoder.calls) == 4
