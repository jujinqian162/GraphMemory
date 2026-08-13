from __future__ import annotations

from collections import Counter
import hashlib
from typing import cast

import pytest
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
    HOMOGENEOUS_PROVENANCE_RELATION_VOCAB,
    PROVENANCE_PHYSICAL_RELATIONS,
    PROVENANCE_RELATION_VOCAB,
    provenance_control_summary,
    provenance_rgcn_model_config,
    tensorize_provenance_edges,
    tensorize_provenance_ranking_task,
    tensorize_provenance_ranking_tasks,
)
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import (
    ExecutionProvenanceRankingRequest,
    TextRankingRequest,
)
from graph_memory.retrieval.signals import (
    SeedSignal,
    seed_signals_from_ranked_nodes,
)
from tests.isetrace_fixtures import isetrace_record


class DeterministicEmbeddingProvider:
    embedding_dim = 4

    def encode_task_nodes(
        self, request: TextRankingRequest, node_ids: list[str]
    ) -> torch.Tensor:
        text_by_id = {
            candidate.item_id: candidate.text for candidate in request.candidates
        }
        text_by_id["q"] = request.query_text
        rows: list[list[float]] = []
        for node_id in node_ids:
            digest = hashlib.sha256(text_by_id[node_id].encode()).digest()
            rows.append([float(value) / 255.0 for value in digest[:4]])
        return torch.tensor(rows, dtype=torch.float32)

    def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
        node_ids = ["q", *(candidate.item_id for candidate in request.candidates)]
        embeddings = self.encode_task_nodes(request, node_ids)
        ranked_nodes = [
            RankedNode(
                node_id=candidate.item_id,
                score=float(embeddings[index] @ embeddings[0]),
            )
            for index, candidate in enumerate(request.candidates, start=1)
        ]
        return seed_signals_from_ranked_nodes(request, ranked_nodes)


def _graph_and_request(*, task_id: str, query_text: str):
    trajectory = adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()),
        source_revision="fixture-revision",
    )
    graph = build_provenance_graph(trajectory)
    request = ExecutionProvenanceRankingRequest(
        task_id=task_id,
        query_text=query_text,
        candidates=provenance_unit_candidates(graph),
        graph=graph,
    )
    return graph, request


def _model_config(*, num_layers: int = 2, ablation_name: str = "full_rgcn"):
    return provenance_rgcn_model_config(
        encoder_model="fake-encoder",
        encoder_dim=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=8,
        hidden_dim=8,
        num_layers=num_layers,
        dropout=0.0,
        ablation_name=ablation_name,
    )


def test_provenance_tensorizer_uses_fixed_bidirectional_physical_relations() -> None:
    graph, request = _graph_and_request(
        task_id="tensor-task", query_text="Which output is relevant?"
    )
    fingerprint = graph.fingerprint()
    edges = tensorize_provenance_edges(graph)
    provider = DeterministicEmbeddingProvider()
    task = tensorize_provenance_ranking_task(
        request,
        model_config=_model_config(),
        text_embedding_provider=provider,
        seed_signal_provider=provider,
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
        assert (
            edges.relation_ids[forward].item()
            == relation_index[f"{edge.relation}_forward"]
        )
        assert (
            edges.relation_ids[reverse].item()
            == relation_index[f"{edge.relation}_reverse"]
        )

    graph_tensor = task.graph_tensor
    assert torch.equal(graph_tensor.edge_index, edges.edge_index)
    assert torch.equal(graph_tensor.relation_ids, edges.relation_ids)
    assert torch.equal(graph_tensor.edge_weights, edges.edge_weights)
    assert graph_tensor.node_ids == [node.node_id for node in graph.nodes] + ["q"]
    assert graph_tensor.query_node_index == len(graph.nodes)
    assert graph_tensor.node_features.shape == (len(graph.nodes) + 1, 1)
    assert task.sample_node_features.shape == (len(request.candidates), 1)
    assert graph_tensor.edge_index.max().item() < len(graph.nodes)
    assert task.sample_node_ids == [
        candidate.item_id for candidate in request.candidates
    ]
    assert all(
        graph_tensor.node_ids[index] == candidate.item_id
        for index, candidate in zip(
            task.sample_node_indices.tolist(), request.candidates, strict=True
        )
    )
    assert graph.fingerprint() == fingerprint


def test_homogeneous_control_preserves_topology_and_collapses_all_relation_ids() -> (
    None
):
    graph, _request = _graph_and_request(task_id="homogeneous", query_text="query")
    full = tensorize_provenance_edges(graph, model_config=_model_config())
    homogeneous_config = _model_config(ablation_name="homogeneous_gcn")
    homogeneous = tensorize_provenance_edges(graph, model_config=homogeneous_config)

    assert homogeneous_config.message_transform_type == "shared"
    assert homogeneous_config.relation_vocab == HOMOGENEOUS_PROVENANCE_RELATION_VOCAB
    assert torch.equal(homogeneous.edge_index, full.edge_index)
    assert torch.equal(homogeneous.edge_weights, full.edge_weights)
    assert homogeneous.relation_ids.tolist() == [0] * len(homogeneous.relation_ids)


@pytest.mark.parametrize(
    ("variant", "removed_relations"),
    (
        ("wo_feeds", {"data.feeds"}),
        (
            "wo_execution_ownership",
            {
                "execution.returns",
                "execution.has_argument",
                "execution.has_content",
            },
        ),
        ("wo_artifact_io", {"resource.reads", "resource.writes"}),
        ("wo_chunk_adjacency", {"content.next"}),
    ),
)
def test_relation_family_control_removes_only_declared_relations(
    variant: str,
    removed_relations: set[str],
) -> None:
    graph, _request = _graph_and_request(task_id=variant, query_text="query")
    full_config = _model_config()
    control_config = _model_config(ablation_name=variant)
    full_summary = provenance_control_summary([graph], model_config=full_config)
    control_summary = provenance_control_summary([graph], model_config=control_config)
    full_counts = Counter(cast(dict[str, int], full_summary["relation_counts"]))
    control_counts = Counter(cast(dict[str, int], control_summary["relation_counts"]))

    assert control_config.relation_vocab == PROVENANCE_RELATION_VOCAB
    assert control_config.message_transform_type == "typed"
    assert (
        set(full_config.enabled_provenance_relations)
        - set(control_config.enabled_provenance_relations)
        == removed_relations
    )
    assert control_counts == Counter(
        {
            relation: count
            for relation, count in full_counts.items()
            if relation not in removed_relations
        }
    )


@pytest.mark.parametrize(
    "variant",
    (
        "homogeneous_gcn",
        "wo_feeds",
        "wo_execution_ownership",
        "wo_artifact_io",
        "wo_chunk_adjacency",
        "random_edges",
    ),
)
def test_relation_control_configs_build_and_score_the_same_candidates(
    variant: str,
) -> None:
    _graph, request = _graph_and_request(task_id=variant, query_text="query")
    provider = DeterministicEmbeddingProvider()
    config = _model_config(ablation_name=variant)
    task = tensorize_provenance_ranking_task(
        request,
        model_config=config,
        text_embedding_provider=provider,
        seed_signal_provider=provider,
    )
    model = build_model_from_config(config)
    model.eval()

    with torch.no_grad():
        scores = model(collate_evidence_tasks([task]))

    assert scores.shape == (len(request.candidates),)
    assert task.sample_node_ids == [
        candidate.item_id for candidate in request.candidates
    ]


def test_random_edge_control_is_deterministic_and_degree_preserving() -> None:
    graph, _request = _graph_and_request(task_id="random-edges", query_text="query")
    fingerprint = graph.fingerprint()
    full = tensorize_provenance_edges(graph, model_config=_model_config())
    random_config = _model_config(ablation_name="random_edges")
    randomized = tensorize_provenance_edges(graph, model_config=random_config)
    repeated = tensorize_provenance_edges(graph, model_config=random_config)
    summary = provenance_control_summary([graph], model_config=random_config)

    def degree_signature(edge_index: torch.Tensor, relation_ids: torch.Tensor):
        result: Counter[tuple[int, str, int]] = Counter()
        for column, relation_id in enumerate(relation_ids.tolist()):
            source, target = edge_index[:, column].tolist()
            result[(relation_id, "out", source)] += 1
            result[(relation_id, "in", target)] += 1
        return result

    assert random_config.message_topology == "degree_preserving_random_v1"
    assert random_config.message_topology_seed == 13
    assert torch.equal(randomized.edge_index, repeated.edge_index)
    assert torch.equal(randomized.relation_ids, full.relation_ids)
    assert not torch.equal(randomized.edge_index, full.edge_index)
    assert degree_signature(randomized.edge_index, randomized.relation_ids) == (
        degree_signature(full.edge_index, full.relation_ids)
    )
    assert cast(int, summary["rewired_edge_count"]) > 0
    assert summary["changed_graph_count"] == 1
    assert cast(int, summary["physical_edge_count"]) >= cast(
        int, summary["active_edge_count"]
    )
    assert graph.fingerprint() == fingerprint


def test_provenance_tasks_collate_without_cross_task_query_or_candidate_ownership() -> (
    None
):
    graph, first = _graph_and_request(
        task_id="first-task", query_text="Find the upstream output."
    )
    second = first.model_copy(
        update={"task_id": "second-task", "query_text": "Find the downstream output."}
    )
    provider = DeterministicEmbeddingProvider()
    tasks = tensorize_provenance_ranking_tasks(
        [first, second],
        model_config=_model_config(),
        text_embedding_provider=provider,
        seed_signal_provider=provider,
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
    assert all(
        index < node_count for index in batch.sample_node_indices[:candidate_count]
    )
    assert all(
        index >= node_count for index in batch.sample_node_indices[candidate_count:]
    )
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
    provider = DeterministicEmbeddingProvider()
    tasks = tensorize_provenance_ranking_tasks(
        [first, second],
        model_config=_model_config(),
        text_embedding_provider=provider,
        seed_signal_provider=provider,
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


def test_zero_layer_provenance_config_reuses_seed_scores_exactly() -> None:
    config = _model_config(num_layers=0)
    model = build_model_from_config(config)
    _graph, request = _graph_and_request(
        task_id="seed-passthrough", query_text="Find the relevant output."
    )
    provider = DeterministicEmbeddingProvider()
    task = tensorize_provenance_ranking_task(
        request,
        model_config=config,
        text_embedding_provider=provider,
        seed_signal_provider=provider,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    batch = collate_evidence_tasks([task])

    assert config.num_layers == 0
    assert config.graph_encoder_type == "identity"
    assert config.scoring_mode == "seed_passthrough"
    assert type(model.graph_encoder).__name__ == "IdentityGraphEncoder"
    assert torch.equal(model(batch), task.sample_node_features[:, 0])
