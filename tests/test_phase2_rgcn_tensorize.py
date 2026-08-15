import torch

from graph_memory.models.graph_retriever.internals.tensorization import (
    EdgeTensorizer,
    UniformEdgeWeightPolicy,
)
from graph_memory.graphs.contracts import EvidenceGraph


def tensor_graph() -> EvidenceGraph:
    return EvidenceGraph.model_validate(
        {
            "task_id": "hotpot_tensor_test",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "question"},
                {
                    "id": "m0",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "first memory",
                    "source_ref": "A",
                    "group_key": "document:A",
                    "sequence_index": 0,
                    "metadata": {"title": "A", "position": 0},
                },
                {
                    "id": "m1",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "second memory",
                    "source_ref": "B",
                    "group_key": "document:B",
                    "sequence_index": 0,
                    "metadata": {"title": "B", "position": 1},
                },
                {
                    "id": "m2",
                    "node_type": "graph_item",
                    "node_kind": "document_sentence",
                    "text": "third memory",
                    "source_ref": "C",
                    "group_key": "document:C",
                    "sequence_index": 0,
                    "metadata": {"title": "C", "position": 2},
                },
            ],
            "edges": [
                {
                    "source": "q",
                    "target": "m0",
                    "edge_type": "query_overlap",
                    "weight": 2.5,
                    "directed": True,
                },
                {
                    "source": "m0",
                    "target": "m1",
                    "edge_type": "bridge",
                    "weight": 0.7,
                    "directed": False,
                },
                {
                    "source": "m1",
                    "target": "m2",
                    "edge_type": "sequential",
                    "weight": 0.3,
                    "directed": True,
                },
            ],
        }
    )


def test_edge_tensorizer_expands_directed_and_undirected_edges():
    tensors = EdgeTensorizer().tensorize_edges(tensor_graph())

    assert tensors.edge_index.tolist() == [
        [0, 1, 2, 2],
        [1, 2, 1, 3],
    ]
    assert tensors.relation_ids.tolist() == [0, 5, 6, 1]
    assert torch.allclose(
        tensors.edge_weights, torch.tensor([2.5, 0.7, 0.7, 0.3], dtype=torch.float32)
    )


def test_edge_tensorizer_filters_disabled_edge_types():
    tensors = EdgeTensorizer(
        enabled_edge_types=frozenset({"query_overlap", "sequential"})
    ).tensorize_edges(tensor_graph())

    assert tensors.edge_index.tolist() == [
        [0, 2],
        [1, 3],
    ]
    assert tensors.relation_ids.tolist() == [0, 1]


def test_uniform_edge_weight_policy_replaces_artifact_weights():
    tensors = EdgeTensorizer(
        edge_weight_policy=UniformEdgeWeightPolicy()
    ).tensorize_edges(tensor_graph())

    assert torch.allclose(tensors.edge_weights, torch.ones(4, dtype=torch.float32))
