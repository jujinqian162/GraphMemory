import numpy as np
import pytest
import graph_memory.retrieval.methods.graph_rerank.engine as rerank_module
import graph_memory.retrieval as retrieval_module
import graph_memory.experiment as experiment_module
from dataclasses import asdict, fields
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.application.run_retrieval import RunRetrievalRequest, run_retrieval as run_retrieval_app
from graph_memory.retrieval.methods.graph_rerank import (
    graph_rerank,
    graph_rerank_with_breakdown,
    neighbor_propagation_scores,
    normalize_scores,
    rank_graph_from_initial_scores,
)
from graph_memory.graphs.views import induced_retrieved_subgraph
from graph_memory.evaluation.service import evaluate_results
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime, TrainableGraphRuntime
from graph_memory.retrieval_registry import (
    METHOD_REGISTRY,
    get_graph_rerank_methods,
    get_methods_requiring_dense_encoder,
    get_supported_methods,
)
from graph_memory.retrieval.methods.graph_rerank.config import (
    GraphRerankConfig,
    TuningCandidateRow,
    ensure_graph_rerank_config,
)
from scripts.run_retrieval import build_parser
from graph_memory.retrieval.tuning import (
    InitialScoreCache,
    graph_rerank_grid,
    graph_rerank_grid_from_record,
    run_graph_rerank_from_initial_score_cache,
    select_best_config,
    tune_graph_rerank as tune_graph_rerank_service,
    tuning_objective,
)
from graph_memory.validation import ContractValidationError, validate_graph_rerank_config


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


class CountingFakeEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts, batch_size=64, normalize_embeddings=True):
        self.encode_calls += 1
        return super().encode(texts, batch_size=batch_size, normalize_embeddings=normalize_embeddings)


def run_retrieval(
    *,
    method,
    task_inputs,
    graphs,
    top_k,
    encoder_model="intfloat/e5-base-v2",
    query_prefix="query: ",
    passage_prefix="passage: ",
    dense_encoder=None,
    graph_config=None,
    checkpoint_path=None,
    text_embedding_provider=None,
    seed_signal_provider=None,
    device="cpu",
):
    return run_retrieval_app(
        RunRetrievalRequest(
            method=method,
            task_inputs=task_inputs,
            graphs=graphs,
            top_k=top_k,
            dense_runtime=DenseRuntime(
                config=DenseConfig(
                    model_name=encoder_model,
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix,
                ),
                encoder=dense_encoder,
            ),
            graph_config=graph_config,
            trainable_runtime=(
                TrainableGraphRuntime(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    text_embedding_provider=text_embedding_provider,
                    seed_signal_provider=seed_signal_provider,
                )
                if checkpoint_path is not None
                else None
            ),
        )
    )


def tune_graph_rerank(
    *,
    method,
    task_inputs,
    labels,
    graphs,
    grid=None,
    top_k=10,
    encoder_model="intfloat/e5-base-v2",
    query_prefix="query: ",
    passage_prefix="passage: ",
    dense_encoder=None,
):
    return tune_graph_rerank_service(
        method=method,
        task_inputs=task_inputs,
        labels=labels,
        graphs=graphs,
        grid=grid,
        top_k=top_k,
        dense_runtime=DenseRuntime(
            config=DenseConfig(
                model_name=encoder_model,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
            ),
            encoder=dense_encoder,
        ),
    )


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


def retrieval_task_labels() -> list[MemoryTaskLabels]:
    return [
        {
            "task_id": "hotpot_ex1",
            "gold_answer": "Seine",
            "gold_evidence_nodes": ["m0", "m1"],
            "gold_dependency_edges": [],
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


def test_retrieval_method_registry_drives_supported_methods_and_cli_choices():
    supported_methods = get_supported_methods()
    parser_method_action = next(action for action in build_parser()._actions if action.dest == "method")

    assert supported_methods == tuple(METHOD_REGISTRY)
    assert get_graph_rerank_methods() == ("bm25_graph_rerank", "dense_graph_rerank")
    assert get_methods_requiring_dense_encoder() == ("dense", "dense_graph_rerank", "dense_rgcn_graph_retriever")
    assert parser_method_action.choices is not None
    assert tuple(parser_method_action.choices) == supported_methods
    assert METHOD_REGISTRY["bm25"].requires_graphs is False
    assert METHOD_REGISTRY["dense"].requires_graph_config is False
    assert METHOD_REGISTRY["bm25_graph_rerank"].requires_graphs is True
    assert METHOD_REGISTRY["dense_graph_rerank"].seed_method == "dense"
    assert METHOD_REGISTRY["dense_rgcn_graph_retriever"].requires_checkpoint is True
    assert not hasattr(retrieval_module, "METHOD_REGISTRY")
    assert not hasattr(retrieval_module, "get_supported_methods")
    assert not hasattr(retrieval_module, "get_graph_rerank_methods")
    assert not hasattr(retrieval_module, "get_methods_requiring_dense_encoder")
    assert not hasattr(experiment_module, "CURRENT_METHODS")
    assert not hasattr(experiment_module, "GRAPH_RERANK_METHODS")


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


def test_flat_methods_accept_missing_graph_inputs():
    for method in ["bm25", "dense"]:
        result = run_retrieval(
            method=method,
            task_inputs=retrieval_task_inputs(),
            graphs=None,
            top_k=2,
            encoder_model="fake-model",
            dense_encoder=FakeEncoder(),
        )

        assert result[0]["method"] == method
        assert result[0]["retrieved_subgraph"]["edges"] == []


def test_graph_pipeline_requires_graph_config_before_processing():
    with pytest.raises(ValueError, match="Graph rerank methods require graph_config"):
        run_retrieval(
            method="bm25_graph_rerank",
            task_inputs=retrieval_task_inputs(),
            graphs=retrieval_graphs(),
            top_k=2,
        )


def test_graph_rerank_config_uses_neighbor_type_weights_as_canonical_field():
    config_fields = {field.name for field in fields(GraphRerankConfig)}

    assert "neighbor_type_weights" in config_fields
    assert "type_weights" not in config_fields

    config = GraphRerankConfig(
        neighbor_type_weights={
            "sequential": 0.3,
            "entity_overlap": 0.7,
            "bridge": 1.0,
        }
    )
    config_record = asdict(config)

    assert "neighbor_type_weights" in config_record
    assert "type_weights" not in config_record
    assert "query_overlap" not in config.neighbor_type_weights
    validate_graph_rerank_config(config_record)


def test_deprecated_type_weights_record_is_rejected():
    with pytest.raises(ValueError, match="type_weights is deprecated; use neighbor_type_weights instead"):
        ensure_graph_rerank_config(
            {
                "lambda_init": 1.0,
                "lambda_query": 0.1,
                "lambda_neighbor": 0.2,
                "lambda_bridge": 0.1,
                "lambda_path": 0.0,
                "seed_top_s": 30,
                "max_hops": 2,
                "type_weights": {
                    "query_overlap": 0.0,
                    "sequential": 0.3,
                    "entity_overlap": 0.7,
                    "bridge": 1.0,
                },
            }
        )


def test_type_weights_is_rejected_even_when_neighbor_type_weights_is_present():
    with pytest.raises(ValueError, match="type_weights is deprecated; use neighbor_type_weights instead"):
        ensure_graph_rerank_config(
            {
                "lambda_init": 1.0,
                "lambda_query": 0.1,
                "lambda_neighbor": 0.2,
                "lambda_bridge": 0.1,
                "lambda_path": 0.0,
                "seed_top_s": 30,
                "max_hops": 2,
                "neighbor_type_weights": {
                    "sequential": 0.1,
                    "entity_overlap": 0.2,
                    "bridge": 0.3,
                },
                "type_weights": {
                    "query_overlap": 99.0,
                    "sequential": 9.0,
                    "entity_overlap": 9.0,
                    "bridge": 9.0,
                },
            }
        )


def test_graph_pipeline_requires_graph_for_every_task():
    task_inputs = retrieval_task_inputs()
    second_task: MemoryTaskInput = {
        **task_inputs[0],
        "task_id": "hotpot_ex2",
    }

    with pytest.raises(ContractValidationError, match="task_id alignment mismatch"):
        run_retrieval(
            method="bm25_graph_rerank",
            task_inputs=[*task_inputs, second_task],
            graphs=retrieval_graphs(),
            top_k=2,
            graph_config={"lambda_init": 1.0, "lambda_query": 0.1, "lambda_neighbor": 0.2, "lambda_bridge": 0.1},
        )


def test_query_overlap_does_not_require_neighbor_type_weight_and_uses_lambda_query_only():
    graph: MemoryGraph = {
        "task_id": "hotpot_ex1",
        "nodes": [],
        "edges": [
            {"source": "q", "target": "m1", "edge_type": "query_overlap", "weight": 5.0, "directed": True},
        ],
    }
    config = GraphRerankConfig(
        lambda_init=0.0,
        lambda_query=1.0,
        lambda_neighbor=0.0,
        lambda_bridge=0.0,
        seed_top_s=2,
        max_hops=1,
        neighbor_type_weights={"sequential": 0.0, "entity_overlap": 0.0, "bridge": 0.0},
    )

    ranked, breakdown = graph_rerank_with_breakdown({"m0": 1.0, "m1": 0.0}, graph, config)

    assert [node.node_id for node in ranked] == ["m1", "m0"]
    assert breakdown["m1"].query == pytest.approx(1.0)
    assert breakdown["m1"].final == pytest.approx(1.0)

    ablated_config = ensure_graph_rerank_config({**asdict(config), "lambda_query": 0.0})
    ablated_ranked = graph_rerank({"m0": 1.0, "m1": 0.0}, graph, ablated_config)

    assert [node.node_id for node in ablated_ranked] == ["m0", "m1"]


def test_query_overlap_component_uses_lambda_query_only():
    config = ensure_graph_rerank_config(
        {
            "lambda_init": 0.0,
            "lambda_query": 1.0,
            "lambda_neighbor": 0.0,
            "lambda_bridge": 0.0,
            "lambda_path": 0.0,
            "seed_top_s": 2,
            "max_hops": 1,
            "neighbor_type_weights": {
                "sequential": 0.0,
                "entity_overlap": 0.0,
                "bridge": 0.0,
            },
        }
    )

    graph: MemoryGraph = {
        "task_id": "hotpot_ex1",
        "nodes": [],
        "edges": [
            {"source": "q", "target": "m1", "edge_type": "query_overlap", "weight": 5.0, "directed": True},
        ],
    }
    ranked = graph_rerank(
        {"m0": 1.0, "m1": 0.0},
        graph,
        config,
    )

    assert [node.node_id for node in ranked] == ["m1", "m0"]


def test_normalize_scores_handles_equal_values():
    assert normalize_scores({"m0": 3.0, "m1": 3.0}) == {"m0": 0.0, "m1": 0.0}


def test_graph_rerank_uses_bridge_to_promote_connected_evidence():
    initial_scores = {"m0": 1.0, "m1": 0.2, "m2": 0.8}
    graph: MemoryGraph = {
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


def test_graph_rerank_normalizes_graph_components_before_combining():
    ranked = graph_rerank(
        {"m0": 1.0, "m1": 0.95},
        {
            "task_id": "hotpot_ex1",
            "nodes": [],
            "edges": [
                {"source": "q", "target": "m1", "edge_type": "query_overlap", "weight": 10_000.0, "directed": True},
                {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 10_000.0, "directed": False},
            ],
        },
        GraphRerankConfig(
            lambda_init=1.0,
            lambda_query=0.2,
            lambda_neighbor=0.0,
            lambda_bridge=0.2,
            seed_top_s=2,
            max_hops=1,
        ),
    )

    assert [node.node_id for node in ranked] == ["m0", "m1"]


def test_retrieval_pipeline_normalizes_graph_components_before_combining():
    task_inputs: list[MemoryTaskInput] = [
        {
            "task_id": "hotpot_ex1",
            "query": "Which city has the landmark?",
            "memory_items": [
                {
                    "id": "m0",
                    "node_type": "document_sentence",
                    "text": "The best evidence sentence.",
                    "source": "Evidence",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
                    "node_type": "document_sentence",
                    "text": "A noisy graph distractor.",
                    "source": "Noise",
                    "sentence_id": 0,
                    "position": 1,
                },
            ],
        }
    ]
    graphs: list[MemoryGraph] = [
        {
            "task_id": "hotpot_ex1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": task_inputs[0]["query"]},
                *task_inputs[0]["memory_items"],
            ],
            "edges": [
                {"source": "q", "target": "m1", "edge_type": "query_overlap", "weight": 10_000.0, "directed": True},
            ],
        }
    ]

    result = run_graph_rerank_from_initial_score_cache(
        method="dense_graph_rerank",
        task_inputs=task_inputs,
        graphs=graphs,
        initial_score_cache=InitialScoreCache(
            scores_by_task_id={"hotpot_ex1": {"m0": 1.0, "m1": 0.95}},
            latency_ms_by_task_id={"hotpot_ex1": 0.0},
        ),
        top_k=1,
        graph_config=GraphRerankConfig(
            lambda_init=1.0,
            lambda_query=0.2,
            lambda_neighbor=0.0,
            lambda_bridge=0.0,
            seed_top_s=2,
            max_hops=1,
        ),
    )

    assert result[0]["ranked_nodes"][0]["node_id"] == "m0"


def test_neighbor_propagation_uses_weighted_average_not_edge_count():
    graph: MemoryGraph = {
        "task_id": "hotpot_ex1",
        "nodes": [],
        "edges": [
            {"source": "m0", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            {"source": "m1", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            {"source": "m2", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            {"source": "m0", "target": "m4", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        ],
    }

    scores = neighbor_propagation_scores(
        {"m0": 1.0, "m1": 1.0, "m2": 1.0, "m3": 0.0, "m4": 0.0},
        graph,
        GraphRerankConfig(),
    )

    assert scores["m3"] == pytest.approx(scores["m4"])
    assert scores["m3"] <= 1.0


def test_induced_retrieved_subgraph_keeps_edges_inside_selected_nodes():
    graph: MemoryGraph = {
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


def test_graph_rerank_retrieval_promotes_bridge_neighbor_and_emits_induced_edge():
    result = run_retrieval(
        method="bm25_graph_rerank",
        task_inputs=retrieval_task_inputs(),
        graphs=retrieval_graphs(),
        top_k=2,
        graph_config=GraphRerankConfig(
            lambda_init=0.0,
            lambda_query=0.1,
            lambda_neighbor=0.0,
            lambda_bridge=1.0,
            seed_top_s=1,
            max_hops=1,
        ),
    )

    assert result[0]["method"] == "bm25_graph_rerank"
    assert len(result[0]["ranked_nodes"]) == 3
    assert {ranked_node["node_id"] for ranked_node in result[0]["ranked_nodes"][:2]} == {"m0", "m1"}
    assert result[0]["retrieved_subgraph"]["edges"] == [
        {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 2.0, "directed": False}
    ]


def test_rerank_entrypoint_matches_retrieval_cache_path_for_ranking_and_edges():
    assert callable(rerank_module.rank_graph_from_initial_scores)

    config = GraphRerankConfig(
        lambda_init=0.0,
        lambda_query=0.1,
        lambda_neighbor=0.0,
        lambda_bridge=1.0,
        seed_top_s=1,
        max_hops=1,
    )
    initial_scores = {"m0": 3.0, "m1": 0.1, "m2": 0.0}
    direct_result = rank_graph_from_initial_scores(initial_scores, retrieval_graphs()[0], config, top_k=2)
    retrieval_result = run_graph_rerank_from_initial_score_cache(
        method="bm25_graph_rerank",
        task_inputs=retrieval_task_inputs(),
        graphs=retrieval_graphs(),
        initial_score_cache=InitialScoreCache(
            scores_by_task_id={"hotpot_ex1": initial_scores},
            latency_ms_by_task_id={"hotpot_ex1": 0.0},
        ),
        top_k=2,
        graph_config=config,
    )[0]

    assert [node.node_id for node in direct_result.ranked_nodes] == [
        record["node_id"] for record in retrieval_result["ranked_nodes"]
    ]
    assert direct_result.retrieved_subgraph == retrieval_result["retrieved_subgraph"]


def test_tuning_objective_weights_full_support_recall_and_connected_evidence():
    row = cast(MetricRow, cast(object, {"Full Support@5": 0.6, "Recall@5": 0.5, "Connected Evidence Recall@10": 0.25}))

    assert tuning_objective(row) == pytest.approx(0.5)


def test_grid_search_selects_highest_objective_then_latency_tiebreak():
    rows = cast(
        list[TuningCandidateRow],
        cast(
            object,
            [
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
            ],
        ),
    )

    assert select_best_config(rows)["lambda_neighbor"] == 0.2


def test_graph_rerank_grid_keeps_lambda_path_zero_for_hotpotqa():
    grid = graph_rerank_grid()

    assert grid
    assert {config.lambda_path for config in grid} == {0.0}


def test_graph_rerank_grid_includes_pure_initial_score_fallback():
    grid = graph_rerank_grid()

    assert any(
        config.lambda_query == 0.0
        and config.lambda_neighbor == 0.0
        and config.lambda_bridge == 0.0
        for config in grid
    )


def test_graph_rerank_grid_from_record_reads_neighbor_type_weights_and_rejects_deprecated_type_weights():
    canonical_grid = graph_rerank_grid_from_record(
        {
            "lambda_init": [1.0],
            "lambda_query": [0.0],
            "lambda_neighbor": [0.1],
            "lambda_bridge": [0.0],
            "lambda_path": [0.0],
            "seed_top_s": [1],
            "max_hops": [1],
            "neighbor_type_weights": {
                "sequential": 0.1,
                "entity_overlap": 0.2,
                "bridge": 0.3,
            },
        }
    )

    assert canonical_grid[0].neighbor_type_weights == {
        "sequential": 0.1,
        "entity_overlap": 0.2,
        "bridge": 0.3,
    }
    with pytest.raises(ValueError, match="type_weights is deprecated; use neighbor_type_weights instead"):
        graph_rerank_grid_from_record(
            {
                "lambda_init": [1.0],
                "lambda_query": [0.0],
                "lambda_neighbor": [0.1],
                "lambda_bridge": [0.0],
                "lambda_path": [0.0],
                "seed_top_s": [1],
                "max_hops": [1],
                "type_weights": {
                    "query_overlap": 99.0,
                    "sequential": 0.1,
                    "entity_overlap": 0.2,
                    "bridge": 0.3,
                },
            }
        )


def test_dense_graph_rerank_tuning_reuses_seed_scores_across_grid():
    encoder = CountingFakeEncoder()
    grid = [
        GraphRerankConfig(lambda_query=0.0, lambda_neighbor=0.05, lambda_bridge=0.0, seed_top_s=1, max_hops=1),
        GraphRerankConfig(lambda_query=0.1, lambda_neighbor=0.05, lambda_bridge=0.0, seed_top_s=1, max_hops=1),
    ]

    tune_graph_rerank(
        method="dense_graph_rerank",
        task_inputs=retrieval_task_inputs(),
        labels=retrieval_task_labels(),
        graphs=retrieval_graphs(),
        grid=grid,
        top_k=2,
        encoder_model="fake-model",
        dense_encoder=encoder,
    )

    assert encoder.encode_calls == 2


def test_tuning_candidate_metrics_match_normal_retrieval_path():
    config = GraphRerankConfig(lambda_query=0.1, lambda_neighbor=0.05, lambda_bridge=0.0, seed_top_s=1, max_hops=1)
    graphs = retrieval_graphs()
    _, candidate_rows = tune_graph_rerank(
        method="dense_graph_rerank",
        task_inputs=retrieval_task_inputs(),
        labels=retrieval_task_labels(),
        graphs=graphs,
        grid=[config],
        top_k=2,
        encoder_model="fake-model",
        dense_encoder=FakeEncoder(),
    )
    expected_rows = evaluate_results(
        run_retrieval(
            method="dense_graph_rerank",
            task_inputs=retrieval_task_inputs(),
            graphs=graphs,
            top_k=2,
            encoder_model="fake-model",
            dense_encoder=FakeEncoder(),
            graph_config=config,
        ),
        retrieval_task_labels(),
        graphs,
    )

    for key, value in expected_rows[0].items():
        if key == "Retrieval Latency / Query":
            continue
        if isinstance(value, float):
            assert candidate_rows[0][key] == pytest.approx(value)
        else:
            assert candidate_rows[0][key] == value


def test_tuning_records_write_neighbor_type_weights_only():
    config = GraphRerankConfig(
        lambda_query=0.1,
        lambda_neighbor=0.05,
        lambda_bridge=0.0,
        seed_top_s=1,
        max_hops=1,
        neighbor_type_weights={"sequential": 0.3, "entity_overlap": 0.7, "bridge": 1.0},
    )

    selected_config, candidate_rows = tune_graph_rerank(
        method="bm25_graph_rerank",
        task_inputs=retrieval_task_inputs(),
        labels=retrieval_task_labels(),
        graphs=retrieval_graphs(),
        grid=[config],
        top_k=2,
    )

    assert "neighbor_type_weights" in selected_config
    assert "type_weights" not in selected_config
    assert "query_overlap" not in selected_config["neighbor_type_weights"]
    assert "neighbor_type_weights" in candidate_rows[0]["config"]
    assert "type_weights" not in candidate_rows[0]["config"]
