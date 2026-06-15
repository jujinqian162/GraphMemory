import numpy as np
import pytest
import graph_memory.retrieval.methods.graph_rerank.engine as rerank_module
import graph_memory.retrieval as retrieval_module
from dataclasses import asdict, fields
from pathlib import Path
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import MetricRow
from graph_memory.contracts.tasks import MemoryTaskInput, MemoryTaskLabels
from graph_memory.registry import Registry
from graph_memory.registry.methods import EncoderSource, GraphConfigSource, GraphInputSource, ModelSource, RetrievalLifecycle
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    DenseEncoderSettings,
    DenseRetrievalSettings,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    RetrievalMethodId,
    SeedRetrievalSettings,
)
from graph_memory.registry.stage_configs import RetrieveIO, RetrieveStageConfig
from graph_memory.stages.retrieve import run_retrieve_stage
from graph_memory.retrieval.methods.graph_rerank.components import neighbor_propagation_scores
from graph_memory.retrieval.methods.graph_rerank.engine import rank_graph_from_initial_scores
from graph_memory.retrieval.methods.graph_rerank.normalization import normalize_scores
from graph_memory.graphs.views import induced_retrieved_subgraph
from graph_memory.evaluation.service import evaluate_results
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.requests import DenseRuntime
from graph_memory.retrieval.methods.graph_rerank.config import (
    GraphRerankConfig,
    ensure_graph_rerank_config,
)
from graph_memory.retrieval.tuning.grid import graph_rerank_grid, graph_rerank_grid_from_record
from graph_memory.retrieval.tuning.initial_scores import (
    InitialScoreCache,
    run_graph_rerank_from_initial_score_cache,
)
from graph_memory.retrieval.tuning.selection import (
    retrieval_candidate_key,
    retrieval_tuning_objective,
)
from graph_memory.retrieval.tuning.service import tune_graph_rerank as tune_graph_rerank_service
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
):
    encoder = DenseEncoderSettings(
        model_name=encoder_model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    method_id = RetrievalMethodId(method)
    if method_id is RetrievalMethodId.BM25:
        job = Bm25RetrievalSettings(top_k=top_k)
    elif method_id is RetrievalMethodId.DENSE:
        job = DenseRetrievalSettings(top_k=top_k, encoder=encoder)
    elif method_id is RetrievalMethodId.BM25_GRAPH_RERANK:
        rerank = ensure_graph_rerank_config(graph_config) if graph_config is not None else GraphRerankConfig()
        job = GraphRerankRetrievalSettings(
            method=method_id,
            top_k=top_k,
            seed=SeedRetrievalSettings(
                method=RetrievalMethodId.BM25,
                encoder=None,
            ),
            rerank=GraphRerankSettings(
                lambda_init=rerank.lambda_init,
                lambda_query=rerank.lambda_query,
                lambda_neighbor=rerank.lambda_neighbor,
                lambda_bridge=rerank.lambda_bridge,
                lambda_path=rerank.lambda_path,
                seed_top_s=rerank.seed_top_s,
                max_hops=rerank.max_hops,
                neighbor_type_weights=dict(rerank.neighbor_type_weights),
            ),
        )
    elif method_id is RetrievalMethodId.DENSE_GRAPH_RERANK:
        rerank = ensure_graph_rerank_config(graph_config) if graph_config is not None else GraphRerankConfig()
        job = GraphRerankRetrievalSettings(
            method=method_id,
            top_k=top_k,
            seed=SeedRetrievalSettings(
                method=RetrievalMethodId.DENSE,
                encoder=encoder,
            ),
            rerank=GraphRerankSettings(
                lambda_init=rerank.lambda_init,
                lambda_query=rerank.lambda_query,
                lambda_neighbor=rerank.lambda_neighbor,
                lambda_bridge=rerank.lambda_bridge,
                lambda_path=rerank.lambda_path,
                seed_top_s=rerank.seed_top_s,
                max_hops=rerank.max_hops,
                neighbor_type_weights=dict(rerank.neighbor_type_weights),
            ),
        )
    else:
        raise ValueError(f"Unsupported test method: {method}")
    config = RetrieveStageConfig(
        io=RetrieveIO(
            tasks=Path("memory_tasks.input.json"),
            graphs=None if graphs is None else Path("graphs.json"),
            output=Path("ranked.json"),
            summary=Path("ranked.run_summary.json"),
            graph_config=None if graph_config is None else Path("graph_config.json"),
        ),
        job=job,
    )
    result = run_retrieve_stage(
        config,
        task_inputs=task_inputs,
        graphs=graphs,
        graph_config=graph_config,
        dense_encoder=dense_encoder,
    )
    return result.predictions


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


def rank_graph_for_test(initial_scores: dict[str, float], graph: MemoryGraph, config: GraphRerankConfig):
    return rank_graph_from_initial_scores(
        initial_scores,
        graph,
        config,
        top_k=len(initial_scores),
    ).ranked_nodes


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
    supported_methods = tuple(method.value for method in Registry.methods.list_ids())

    assert supported_methods == tuple(method.value for method in RetrievalMethodId)
    assert tuple(
        method.value
        for method in Registry.methods.list_by_lifecycle(RetrievalLifecycle.GRAPH_RERANK)
    ) == ("bm25_graph_rerank", "dense_graph_rerank")
    assert Registry.methods.get("bm25").dependencies.graphs is GraphInputSource.NONE
    assert Registry.methods.get("dense").dependencies.graph_config is GraphConfigSource.NONE
    assert Registry.methods.get("dense").dependencies.encoder is EncoderSource.EXPERIMENT_CONFIG
    assert Registry.methods.get("bm25_graph_rerank").dependencies.graphs is GraphInputSource.GRAPH_ARTIFACT
    assert Registry.methods.get("dense_graph_rerank").seed_method is RetrievalMethodId.DENSE
    assert Registry.methods.get("dense_rgcn_graph_retriever").dependencies.model is ModelSource.CHECKPOINT_FILE
    assert not hasattr(retrieval_module, "METHOD_REGISTRY")
    assert not hasattr(retrieval_module, "get_supported_methods")
    assert not hasattr(retrieval_module, "get_graph_rerank_methods")
    assert not hasattr(retrieval_module, "get_methods_requiring_dense_encoder")


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

    ranked = rank_graph_for_test({"m0": 1.0, "m1": 0.0}, graph, config)

    assert [node.node_id for node in ranked] == ["m1", "m0"]

    ablated_config = ensure_graph_rerank_config({**asdict(config), "lambda_query": 0.0})
    ablated_ranked = rank_graph_for_test({"m0": 1.0, "m1": 0.0}, graph, ablated_config)

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
    ranked = rank_graph_for_test(
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

    ranked = rank_graph_for_test(initial_scores, graph, config)

    assert {node.node_id for node in ranked[:2]} == {"m0", "m2"}


def test_graph_rerank_returns_all_original_nodes():
    ranked = rank_graph_for_test(
        {"m0": 1.0, "m1": 0.5, "m2": 0.0},
        {"task_id": "hotpot_ex1", "nodes": [], "edges": []},
        GraphRerankConfig(seed_top_s=1, max_hops=1),
    )

    assert {node.node_id for node in ranked} == {"m0", "m1", "m2"}


def test_graph_rerank_normalizes_graph_components_before_combining():
    ranked = rank_graph_for_test(
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

    assert retrieval_tuning_objective(row) == pytest.approx(0.5)


def test_retrieval_candidate_key_uses_full_support_10_after_objective() -> None:
    lower = _tuning_metric_row(full_support_10=0.6)
    higher = _tuning_metric_row(full_support_10=0.7)

    assert retrieval_candidate_key(higher) > retrieval_candidate_key(lower)


def test_retrieval_candidate_key_prefers_lower_latency_after_support() -> None:
    slower = _tuning_metric_row(latency=20.0)
    faster = _tuning_metric_row(latency=10.0)

    assert retrieval_candidate_key(faster) > retrieval_candidate_key(slower)


def test_retrieval_candidate_key_prefers_fewer_edges_after_latency() -> None:
    more_edges = _tuning_metric_row(avg_edges=6.0)
    fewer_edges = _tuning_metric_row(avg_edges=5.0)

    assert retrieval_candidate_key(fewer_edges) > retrieval_candidate_key(more_edges)


def _tuning_metric_row(
    *,
    full_support_10: float = 0.7,
    latency: float = 10.0,
    avg_edges: float = 5.0,
) -> MetricRow:
    return cast(
        MetricRow,
        cast(
            object,
            {
                "Full Support@5": 0.5,
                "Full Support@10": full_support_10,
                "Recall@5": 0.5,
                "Connected Evidence Recall@10": 0.5,
                "Retrieval Latency / Query": latency,
                "Avg Retrieved Edges": avg_edges,
            },
        ),
    )


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

    assert encoder.encode_calls == 1


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
