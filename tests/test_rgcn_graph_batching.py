from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from graph_memory.contracts.common import TrainPairSampleType
from graph_memory.graphs.contracts import EvidenceGraph
from graph_memory.training_pairs.contracts import TrainPairRecord
from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.models.frozen_embeddings import (
    FrozenTaskEmbeddings,
    node_ids_digest,
)
from graph_memory.models.graph_batching import (
    TaskGraphTensor,
    collate_task_graphs,
    move_graph_batch,
    validate_graph_batch,
)
from graph_memory.models.graph_retriever.batching import (
    build_evidence_dataloader,
    materialize_training_tasks,
)
from graph_memory.models.graph_retriever.text_embeddings import (
    PrecomputedGraphFeatureProvider,
)
from graph_memory.models.graph_retriever.training import train_graph_retriever
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import RetrieverSeedSignalProvider
from tests.rgcn_fixtures import (
    FakeRetriever,
    FakeTextEmbeddingProvider,
    tiny_graphs,
    tiny_model_config,
    tiny_task_inputs,
    tiny_training_config,
)


def test_shared_collator_offsets_queries_edges_and_rejects_cross_task_edges() -> None:
    first = _graph_task("first", node_count=2, query_index=0)
    second = _graph_task("second", node_count=3, query_index=1)

    batch = collate_task_graphs([first, second])

    assert batch.task_node_offsets == [0, 2, 5]
    assert batch.query_node_indices.tolist() == [0, 3]
    assert batch.edge_index.tolist() == [[0, 2], [1, 4]]
    assert batch.task_ids == ["first", "second"]
    validate_graph_batch(batch)
    moved = move_graph_batch(batch, "cpu")
    assert moved.task_node_offsets is batch.task_node_offsets
    assert moved.task_ids is batch.task_ids
    assert moved.node_ids_by_task is batch.node_ids_by_task
    assert moved.node_embeddings.device.type == "cpu"

    leaking = replace(
        batch,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        relation_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_weights=torch.ones(2),
    )
    with pytest.raises(ValueError, match="cross-task edge"):
        validate_graph_batch(leaking)


def test_evidence_dataloader_materializes_features_once_and_keeps_tail_batch() -> None:
    requests, graphs, pairs, _labels = _evidence_tasks([2, 1, 3])
    provider = _CountingEmbeddingProvider()
    tasks = materialize_training_tasks(
        ranking_requests=requests,
        graphs=graphs,
        pairs=pairs,
        model_config=tiny_model_config(),
        text_embedding_provider=provider,
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
    )
    calls_after_materialization = provider.calls
    first_loader = build_evidence_dataloader(
        tasks,
        per_device_graph_batch_size=2,
        shuffle=True,
        random_seed=13,
    )
    second_loader = build_evidence_dataloader(
        tasks,
        per_device_graph_batch_size=2,
        shuffle=True,
        random_seed=13,
    )

    first_epochs = [
        [batch.graph_batch.task_ids for batch in first_loader] for _ in range(2)
    ]
    second_epochs = [
        [batch.graph_batch.task_ids for batch in second_loader] for _ in range(2)
    ]

    assert first_epochs == second_epochs
    assert [len(batch_ids) for batch_ids in first_epochs[0]] == [2, 1]
    assert first_epochs[0] != first_epochs[1]
    assert provider.calls == calls_after_materialization


def test_evidence_tensorization_accepts_precomputed_frozen_embeddings() -> None:
    requests, graphs, pairs, _labels = _evidence_tasks([2])
    node_ids = [node.id for node in graphs[0].nodes]
    values = FakeTextEmbeddingProvider().encode_task_nodes(requests[0], node_ids)
    frozen = FrozenTaskEmbeddings(
        task_id=requests[0].task_id,
        node_ids_digest=node_ids_digest(node_ids),
        values=values,
    )
    provider = PrecomputedGraphFeatureProvider(
        {requests[0].task_id: frozen}, embedding_dim=values.shape[1]
    )

    tasks = materialize_training_tasks(
        ranking_requests=requests,
        graphs=graphs,
        pairs=pairs,
        model_config=tiny_model_config(),
        text_embedding_provider=provider,
        seed_signal_provider=provider,
    )

    assert len(tasks) == 1
    assert torch.equal(tasks[0].graph_tensor.node_embeddings, values)


def test_evidence_tail_graph_batch_produces_its_own_optimizer_step() -> None:
    requests, graphs, pairs, labels = _evidence_tasks([2, 1, 3])
    result = train_graph_retriever(
        train_requests=requests,
        train_graphs=graphs,
        train_labels=labels,
        train_pairs=pairs,
        dev_requests=requests,
        dev_labels=labels,
        dev_graphs=graphs,
        model_config=tiny_model_config(),
        text_embedding_provider=FakeTextEmbeddingProvider(),
        seed_signal_provider=RetrieverSeedSignalProvider(FakeRetriever()),
        device="cpu",
        training_config=tiny_training_config().model_copy(
            update={"epochs": 1, "per_device_graph_batch_size": 2}
        ),
    )

    metrics = result.metric_records[0]
    assert result.global_step == 2
    assert metrics["train_optimizer_step_count"] == 2
    assert metrics["actual_tasks_per_optimizer_step"] == [2, 1]
    assert metrics["train_supervised_sample_count"] == 6
    assert cast(int, metrics["materialized_cpu_tensor_bytes"]) > 0
    assert metrics["train_nodes_per_task_max"] == 4
    assert cast(float, metrics["train_tasks_per_second"]) > 0.0


class _CountingEmbeddingProvider(FakeTextEmbeddingProvider):
    def __init__(self) -> None:
        self.calls: int = 0

    def encode_task_nodes(
        self, request: TextRankingRequest, node_ids: list[str]
    ) -> torch.Tensor:
        self.calls += 1
        return super().encode_task_nodes(request, node_ids)


def _graph_task(task_id: str, *, node_count: int, query_index: int) -> TaskGraphTensor:
    return TaskGraphTensor(
        node_embeddings=torch.arange(node_count * 2, dtype=torch.float32).reshape(
            node_count, 2
        ),
        node_features=torch.zeros((node_count, 1)),
        edge_index=torch.tensor([[0], [node_count - 1]], dtype=torch.long),
        relation_ids=torch.tensor([0], dtype=torch.long),
        edge_weights=torch.tensor([1.0]),
        query_node_index=query_index,
        task_id=task_id,
        node_ids=[f"{task_id}-{index}" for index in range(node_count)],
    )


def _evidence_tasks(
    sample_counts: list[int],
) -> tuple[
    list[TextRankingRequest],
    list[EvidenceGraph],
    list[TrainPairRecord],
    list[EvidenceLabel],
]:
    base_task = tiny_task_inputs()[0]
    base_graph = tiny_graphs()[0]
    requests: list[TextRankingRequest] = []
    graphs: list[EvidenceGraph] = []
    pairs: list[TrainPairRecord] = []
    labels: list[EvidenceLabel] = []
    rows: tuple[tuple[str, int, TrainPairSampleType], ...] = (
        ("m0", 1, "positive"),
        ("m1", 0, "easy_random"),
        ("m2", 0, "hard_bm25"),
    )
    for index, sample_count in enumerate(sample_counts):
        task_id = f"evidence-batch-{index}"
        task = base_task.model_copy(update={"task_id": task_id})
        graph = base_graph.model_copy(update={"task_id": task_id})
        requests.extend(text_ranking_requests_for_dataset("hotpotqa", [task]))
        graphs.append(graph)
        labels.append(
            EvidenceLabel(
                task_id=task_id,
                gold_answer="Alpha",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(),
            )
        )
        for node_id, row_label, sample_type in rows[:sample_count]:
            pairs.append(
                TrainPairRecord(
                    task_id=task_id,
                    node_id=node_id,
                    label=1 if row_label == 1 else 0,
                    sample_type=sample_type,
                )
            )
    return requests, graphs, pairs, labels
