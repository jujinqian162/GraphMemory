from __future__ import annotations

import hashlib

import torch

from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.datasets.isetrace.retrieval_views import provenance_unit_candidates
from graph_memory.graphs.provenance import PRECEDES_EDGE, build_provenance_graph
from graph_memory.models.graph_batching import validate_graph_batch
from graph_memory.models.graph_retriever.batching import (
    build_evidence_dataloader,
    collate_evidence_tasks,
    move_training_batch,
    split_batch_node_scores,
)
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.provenance import (
    PROVENANCE_PHYSICAL_RELATIONS,
    PROVENANCE_RELATION_VOCAB,
    provenance_rgcn_model_config,
    tensorize_provenance_edges,
    tensorize_provenance_ranking_task,
    tensorize_provenance_ranking_tasks,
)
from graph_memory.retrieval.requests import ProvenanceRgcnRequest, TextRankingRequest
from tests.isetrace_fixtures import isetrace_record


class DeterministicEmbeddingProvider:
    embedding_dim = 4

    def encode_task_nodes(
        self, request: TextRankingRequest, node_ids: list[str]
    ) -> torch.Tensor:
        text_by_id = {candidate.item_id: candidate.text for candidate in request.candidates}
        text_by_id["q"] = request.query_text
        rows: list[list[float]] = []
        for node_id in node_ids:
            digest = hashlib.sha256(text_by_id[node_id].encode()).digest()
            rows.append([float(value) / 255.0 for value in digest[:4]])
        return torch.tensor(rows, dtype=torch.float32)


def _graph_and_request(*, task_id: str, query_text: str):
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()),
        source_revision="fixture-revision",
    )
    graph = build_provenance_graph(trajectory)
    request = ProvenanceRgcnRequest(
        task_id=task_id,
        query_text=query_text,
        candidates=provenance_unit_candidates(graph),
        graph=graph,
    )
    return graph, request


def _model_config(*, num_layers: int = 2):
    return provenance_rgcn_model_config(
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=8,
        hidden_dim=8,
        num_layers=num_layers,
        dropout=0.0,
    )


def test_provenance_tensorizer_uses_fixed_bidirectional_physical_relations() -> None:
    graph, request = _graph_and_request(
        task_id="tensor-task", query_text="Which output is relevant?"
    )
    fingerprint = graph.fingerprint()
    edges = tensorize_provenance_edges(graph)
    task = tensorize_provenance_ranking_task(
        request,
        model_config=_model_config(),
        text_embedding_provider=DeterministicEmbeddingProvider(),
    )

    enabled_edges = [
        edge for edge in graph.edges if edge.relation in PROVENANCE_PHYSICAL_RELATIONS
    ]
    assert len(enabled_edges) * 2 == edges.edge_index.shape[1]
    assert edges.relation_ids.shape == (len(enabled_edges) * 2,)
    assert int(edges.relation_ids.min()) >= 0
    assert int(edges.relation_ids.max()) < len(PROVENANCE_RELATION_VOCAB)
    assert torch.equal(edges.edge_weights, torch.ones(len(enabled_edges) * 2))
    assert any(edge.relation == PRECEDES_EDGE for edge in graph.edges)

    node_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    relation_index = {
        relation: index for index, relation in enumerate(PROVENANCE_RELATION_VOCAB)
    }
    for edge_index, edge in enumerate(enabled_edges):
        forward = edge_index * 2
        reverse = forward + 1
        assert edges.edge_index[:, forward].tolist() == [
            node_index[edge.source],
            node_index[edge.target],
        ]
        assert edges.edge_index[:, reverse].tolist() == [
            node_index[edge.target],
            node_index[edge.source],
        ]
        assert edges.relation_ids[forward].item() == relation_index[
            f"{edge.relation}_forward"
        ]
        assert edges.relation_ids[reverse].item() == relation_index[
            f"{edge.relation}_reverse"
        ]

    graph_tensor = task.graph_tensor
    assert graph_tensor.node_ids == [node.node_id for node in graph.nodes] + ["q"]
    assert graph_tensor.query_node_index == len(graph.nodes)
    assert graph_tensor.node_features.shape == (len(graph.nodes) + 1, 0)
    assert graph_tensor.edge_index.max().item() < len(graph.nodes)
    assert task.sample_node_ids == [candidate.item_id for candidate in request.candidates]
    assert all(
        graph_tensor.node_ids[index] == candidate.item_id
        for index, candidate in zip(
            task.sample_node_indices.tolist(), request.candidates, strict=True
        )
    )
    assert graph.fingerprint() == fingerprint
    assert not hasattr(request, "query_origin")


def test_provenance_tasks_collate_without_cross_task_query_or_candidate_ownership() -> (
    None
):
    graph, first = _graph_and_request(
        task_id="first-task", query_text="Find the upstream output."
    )
    second = first.model_copy(
        update={"task_id": "second-task", "query_text": "Find the downstream output."}
    )
    tasks = tensorize_provenance_ranking_tasks(
        [first, second],
        model_config=_model_config(),
        text_embedding_provider=DeterministicEmbeddingProvider(),
    )
    loader = build_evidence_dataloader(
        tasks,
        per_device_graph_batch_size=2,
        shuffle=False,
        random_seed=13,
    )
    batch = next(iter(loader))
    moved = move_training_batch(batch, "cpu")

    validate_graph_batch(batch.graph_batch)
    assert moved.graph_batch.task_ids is batch.graph_batch.task_ids
    assert moved.sample_task_ids is batch.sample_task_ids
    assert moved.sample_node_ids is batch.sample_node_ids
    node_count = len(graph.nodes) + 1
    assert batch.graph_batch.task_node_offsets == [0, node_count, node_count * 2]
    assert batch.graph_batch.query_node_indices.tolist() == [
        len(graph.nodes),
        node_count + len(graph.nodes),
    ]
    candidate_count = len(first.candidates)
    assert batch.sample_task_ids == [
        *("first-task" for _ in range(candidate_count)),
        *("second-task" for _ in range(candidate_count)),
    ]
    assert all(index < node_count for index in batch.sample_node_indices[:candidate_count])
    assert all(index >= node_count for index in batch.sample_node_indices[candidate_count:])
    assert batch.sample_query_indices[:candidate_count].unique().tolist() == [
        len(graph.nodes)
    ]
    assert batch.sample_query_indices[candidate_count:].unique().tolist() == [
        node_count + len(graph.nodes)
    ]


def test_dropout_zero_provenance_batch_matches_individual_task_scores() -> None:
    _graph, first = _graph_and_request(
        task_id="first-task", query_text="Find the upstream output."
    )
    second = first.model_copy(
        update={"task_id": "second-task", "query_text": "Find the downstream output."}
    )
    tasks = tensorize_provenance_ranking_tasks(
        [first, second],
        model_config=_model_config(),
        text_embedding_provider=DeterministicEmbeddingProvider(),
    )
    torch.manual_seed(13)
    model = build_model_from_config(_model_config())
    model.eval()

    individual: dict[str, list[tuple[str, float]]] = {}
    with torch.no_grad():
        for task in tasks:
            batch = collate_evidence_tasks([task])
            individual.update(split_batch_node_scores(batch, model(batch)))
        combined_batch = collate_evidence_tasks(tasks)
        combined = split_batch_node_scores(combined_batch, model(combined_batch))

    assert set(combined) == {"first-task", "second-task"}
    for task_id in combined:
        assert [node_id for node_id, _score in combined[task_id]] == [
            node_id for node_id, _score in individual[task_id]
        ]
        assert torch.allclose(
            torch.tensor([score for _node_id, score in combined[task_id]]),
            torch.tensor([score for _node_id, score in individual[task_id]]),
            atol=1e-6,
        )


def test_zero_layer_provenance_config_reuses_identity_graph_encoder() -> None:
    config = _model_config(num_layers=0)
    model = build_model_from_config(config)

    assert config.num_layers == 0
    assert config.graph_encoder_type == "identity"
    assert type(model.graph_encoder).__name__ == "IdentityGraphEncoder"
