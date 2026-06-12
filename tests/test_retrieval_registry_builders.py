from __future__ import annotations

from pathlib import Path

from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    DenseEncoderSettings,
    DenseRetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    RetrievalMethodId,
    SeedRetrievalSettings,
)
from graph_memory.retrieval.execution.service import run_retrieval
from tests.test_phase1_real_retrieval import FakeEncoder, retrieval_graphs, retrieval_task_inputs


def test_registry_builds_bm25_method_from_settings_without_dense_fields() -> None:
    settings = Bm25RetrievalSettings(top_k=2)

    built = Registry.retrieval.build(settings, FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs()))
    predictions = run_retrieval(retrieval_method=built.method, task_inputs=retrieval_task_inputs(), top_k=settings.top_k)

    assert not hasattr(settings, "encoder")
    assert built.method.name == "bm25"
    assert predictions[0]["method"] == "bm25"


def test_registry_builds_dense_method_from_settings_encoder() -> None:
    settings = DenseRetrievalSettings(
        top_k=2,
        encoder=DenseEncoderSettings(model_name="fake-model", query_prefix="query: ", passage_prefix="passage: "),
    )

    built = Registry.retrieval.build(
        settings,
        FlatRetrievalBuildPayload(task_inputs=retrieval_task_inputs(), dense_encoder=FakeEncoder()),
    )
    predictions = run_retrieval(retrieval_method=built.method, task_inputs=retrieval_task_inputs(), top_k=settings.top_k)

    assert built.method.name == "dense"
    assert predictions[0]["method"] == "dense"


def test_registry_builds_graph_rerank_method_from_seed_settings() -> None:
    settings = GraphRerankRetrievalSettings(
        method=RetrievalMethodId.BM25_GRAPH_RERANK,
        top_k=2,
        seed=SeedRetrievalSettings(method=RetrievalMethodId.BM25),
        rerank=GraphRerankSettings(lambda_init=0.0, lambda_query=0.1, lambda_bridge=1.0, seed_top_s=1, max_hops=1),
    )

    built = Registry.retrieval.build(
        settings,
        GraphRerankBuildPayload(task_inputs=retrieval_task_inputs(), graphs=retrieval_graphs()),
    )
    predictions = run_retrieval(retrieval_method=built.method, task_inputs=retrieval_task_inputs(), top_k=settings.top_k)

    assert built.method.name == "bm25_graph_rerank"
    assert predictions[0]["method"] == "bm25_graph_rerank"
    assert predictions[0]["retrieved_subgraph"]["edges"] == [
        {"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 2.0, "directed": False}
    ]


def test_retrieve_stage_uses_registry_dispatch_not_legacy_application_or_factory() -> None:
    stage_source = Path("graph_memory/stages/retrieve.py").read_text(encoding="utf-8")
    script_source = Path("scripts/run_retrieval.py").read_text(encoding="utf-8")

    assert "Registry.retrieval.build(" in stage_source
    assert "run_retrieve_stage(" in script_source
    for source in (stage_source, script_source):
        assert "RunRetrievalRequest" not in source
        assert "resolve_method_build_request" not in source
        assert "build_retrieval_method" not in source
        assert "RetrievalMethodResolveRequest" not in source


def test_legacy_resolver_and_factory_modules_are_removed() -> None:
    legacy_paths = [
        Path("graph_memory/retrieval/resolver.py"),
        Path("graph_memory/retrieval/factory.py"),
    ]

    assert [str(path) for path in legacy_paths if path.exists()] == []
