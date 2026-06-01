import torch

from graph_memory.learned.tensorize import DEFAULT_RELATION_VOCAB, EdgeTensorizer, UniformEdgeWeightPolicy
from graph_memory.learned.training import default_model_config
from graph_memory.types import MemoryGraph


def tensor_graph() -> MemoryGraph:
    return {
        "task_id": "hotpot_tensor_test",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "question"},
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "first memory",
                "source": "A",
                "sentence_id": 0,
                "position": 0,
            },
            {
                "id": "m1",
                "node_type": "document_sentence",
                "text": "second memory",
                "source": "B",
                "sentence_id": 0,
                "position": 1,
            },
            {
                "id": "m2",
                "node_type": "document_sentence",
                "text": "third memory",
                "source": "C",
                "sentence_id": 0,
                "position": 2,
            },
        ],
        "edges": [
            {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 2.5, "directed": True},
            {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 0.7, "directed": False},
            {"source": "m1", "target": "m2", "edge_type": "sequential", "weight": 0.3, "directed": True},
        ],
    }


def test_default_relation_vocab_is_stable():
    assert DEFAULT_RELATION_VOCAB == (
        "query_overlap_forward",
        "sequential_forward",
        "sequential_reverse",
        "entity_overlap_forward",
        "entity_overlap_reverse",
        "bridge_forward",
        "bridge_reverse",
    )


def test_edge_tensorizer_expands_directed_and_undirected_edges():
    tensors = EdgeTensorizer().tensorize_edges(tensor_graph())

    assert tensors.edge_index.tolist() == [
        [0, 1, 2, 2],
        [1, 2, 1, 3],
    ]
    assert tensors.relation_ids.tolist() == [0, 5, 6, 1]
    assert torch.allclose(tensors.edge_weights, torch.tensor([2.5, 0.7, 0.7, 0.3], dtype=torch.float32))


def test_edge_tensorizer_filters_disabled_edge_types():
    tensors = EdgeTensorizer(enabled_edge_types=frozenset({"query_overlap", "sequential"})).tensorize_edges(
        tensor_graph()
    )

    assert tensors.edge_index.tolist() == [
        [0, 2],
        [1, 3],
    ]
    assert tensors.relation_ids.tolist() == [0, 1]


def test_edge_view_ablation_model_configs_remove_exactly_one_visible_edge_type():
    expected_enabled_edge_types = {
        "wo_bridge": {"entity_overlap", "query_overlap", "sequential"},
        "wo_entity_overlap": {"bridge", "query_overlap", "sequential"},
        "wo_sequential": {"bridge", "entity_overlap", "query_overlap"},
        "wo_query_overlap": {"bridge", "entity_overlap", "sequential"},
    }

    for ablation_name, expected in expected_enabled_edge_types.items():
        config = default_model_config(
            encoder_model="fake-encoder",
            encoder_dim=4,
            query_prefix="query: ",
            passage_prefix="passage: ",
            ablation_name=ablation_name,
        )

        assert set(config.enabled_edge_types) == expected
        assert config.ablation_name == ablation_name


def test_uniform_edge_weight_policy_replaces_artifact_weights():
    tensors = EdgeTensorizer(edge_weight_policy=UniformEdgeWeightPolicy()).tensorize_edges(tensor_graph())

    assert torch.allclose(tensors.edge_weights, torch.ones(4, dtype=torch.float32))
