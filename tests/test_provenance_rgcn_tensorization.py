from __future__ import annotations

from collections import Counter
import hashlib
from typing import cast

import pytest
import torch

from graph_memory.datasets.isetrace import adapt_isetrace_record, parse_isetrace_record
from graph_memory.datasets.isetrace.retrieval_views import provenance_unit_candidates
from graph_memory.graphs.provenance import PRECEDES_EDGE, build_provenance_graph
from graph_memory.models.graph_retriever.batching import (
    collate_evidence_tasks,
)
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.provenance import (
    PROVENANCE_PHYSICAL_RELATIONS,
    PROVENANCE_RELATION_VOCAB,
    provenance_control_summary,
    provenance_rgcn_model_config,
    tensorize_provenance_edges,
    tensorize_provenance_ranking_task,
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


@pytest.mark.parametrize(
    ("variant", "removed_relations"),
    (("wo_feeds", {"data.feeds"}),),
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
