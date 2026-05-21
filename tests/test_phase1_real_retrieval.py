import numpy as np
import pytest

from graph_memory.rerank import graph_rerank, induced_retrieved_subgraph, normalize_scores
from graph_memory.retrieval import run_retrieval
from graph_memory.tuning import graph_rerank_grid, select_best_config, tuning_objective
from graph_memory.types import GraphRerankConfig, MemoryGraph, MemoryTaskInput


class FakeEncoder:
    def encode(self, texts, batch_size=64, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "eiffel" in lowered else 0.0,
                    1.0 if "paris" in lowered else 0.0,
                    1.0 if "seine" in lowered else 0.0,
                ]
            )
        array = np.array(vectors, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            array = array / norms
        return array


def retrieval_task_inputs() -> list[MemoryTaskInput]:
    return [
        {
            "task_id": "hotpot_ex1",
            "query": "Which river runs through the city with the Eiffel Tower?",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "The Eiffel Tower is in Paris.",
                    "source": "Eiffel Tower",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
                    "node_type": "document_sentence",
                    "text": "The Seine runs through Paris.",
                    "source": "Paris",
                    "sentence_id": 0,
                    "position": 1,
                },
                {
                    "id": "m2",
                    "node_type": "document_sentence",
                    "text": "Mount Everest is tall.",
                    "source": "Mount Everest",
                    "sentence_id": 0,
                    "position": 2,
                },
            ],
        }
    ]


def retrieval_graphs() -> list[MemoryGraph]:
    task_input = retrieval_task_inputs()[0]
    return [
        {
            "task_id": "hotpot_ex1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": task_input["query"]},
                *task_input["memory_items"],
            ],
            "edges": [
                {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 1.0, "directed": True},
                {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 2.0, "directed": False},
            ],
        }
    ]


def test_bm25_and_dense_emit_same_ranked_schema():
    for method in ["bm25", "dense"]:
        result = run_retrieval(
            method=method,
            task_inputs=retrieval_task_inputs(),
            graphs=[],
            top_k=2,
            encoder_model="fake-model",
            dense_encoder=FakeEncoder(),
        )

        assert result[0]["method"] == method
        assert len(result[0]["ranked_nodes"]) == len(retrieval_task_inputs()[0]["memory_items"])
        assert "node_id" in result[0]["ranked_nodes"][0]
        assert "score" in result[0]["ranked_nodes"][0]
        assert len(result[0]["retrieved_subgraph"]["nodes"]) <= 2
        assert result[0]["retrieved_subgraph"]["edges"] == []


def test_normalize_scores_handles_equal_values():
    assert normalize_scores({"m0": 3.0, "m1": 3.0}) == {"m0": 0.0, "m1": 0.0}


def test_graph_rerank_uses_bridge_to_promote_connected_evidence():
    initial_scores = {"m0": 1.0, "m1": 0.2, "m2": 0.8}
    graph = {
        "task_id": "hotpot_ex1",
        "nodes": [],
        "edges": [
            {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 2.0, "directed": False}
        ],
    }
    config = GraphRerankConfig(
        lambda_init=1.0,
        lambda_neighbor=0.2,
        lambda_bridge=0.2,
        lambda_query=0.0,
        lambda_path=0.0,
        seed_top_s=2,
        max_hops=1,
    )

    ranked = graph_rerank(initial_scores, graph, config)

    assert {node.node_id for node in ranked[:2]} == {"m0", "m2"}


def test_graph_rerank_returns_all_original_nodes():
    ranked = graph_rerank(
        {"m0": 1.0, "m1": 0.5, "m2": 0.0},
        {"task_id": "hotpot_ex1", "nodes": [], "edges": []},
        GraphRerankConfig(seed_top_s=1, max_hops=1),
    )

    assert {node.node_id for node in ranked} == {"m0", "m1", "m2"}


def test_induced_retrieved_subgraph_keeps_edges_inside_selected_nodes():
    graph = {
        "task_id": "hotpot_ex1",
        "nodes": [],
        "edges": [
            {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False},
            {"source": "m1", "target": "m2", "edge_type": "bridge", "weight": 1.0, "directed": False},
        ],
    }

    subgraph = induced_retrieved_subgraph(graph, ["m0", "m1"])

    assert subgraph["nodes"] == ["m0", "m1"]
    assert subgraph["edges"] == [
        {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False}
    ]


def test_graph_rerank_methods_emit_ranked_schema_with_induced_edges():
    result = run_retrieval(
        method="bm25_graph_rerank",
        task_inputs=retrieval_task_inputs(),
        graphs=retrieval_graphs(),
        top_k=2,
        graph_config={"lambda_init": 1.0, "lambda_query": 0.1, "lambda_neighbor": 0.2, "lambda_bridge": 0.1},
    )

    assert result[0]["method"] == "bm25_graph_rerank"
    assert len(result[0]["ranked_nodes"]) == 3
    assert result[0]["retrieved_subgraph"]["edges"]


def test_tuning_objective_weights_full_support_recall_and_connected_evidence():
    row = {"Full Support@5": 0.6, "Recall@5": 0.5, "Connected Evidence Recall@10": 0.25}

    assert tuning_objective(row) == pytest.approx(0.5)


def test_grid_search_selects_highest_objective_then_latency_tiebreak():
    rows = [
        {
            "config": {"lambda_neighbor": 0.1},
            "Full Support@5": 0.5,
            "Full Support@10": 0.7,
            "Recall@5": 0.5,
            "Connected Evidence Recall@10": 0.5,
            "Retrieval Latency / Query": 20.0,
            "Avg Retrieved Edges": 5.0,
        },
        {
            "config": {"lambda_neighbor": 0.2},
            "Full Support@5": 0.5,
            "Full Support@10": 0.7,
            "Recall@5": 0.5,
            "Connected Evidence Recall@10": 0.5,
            "Retrieval Latency / Query": 10.0,
            "Avg Retrieved Edges": 5.0,
        },
    ]

    assert select_best_config(rows)["lambda_neighbor"] == 0.2


def test_graph_rerank_grid_keeps_lambda_path_zero_for_hotpotqa():
    grid = graph_rerank_grid()

    assert grid
    assert {config.lambda_path for config in grid} == {0.0}
