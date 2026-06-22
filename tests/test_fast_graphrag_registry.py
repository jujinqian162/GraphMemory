from __future__ import annotations

from graph_memory.registry.methods import build_method_registry
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    FastGraphRAGBuildPayload,
    FastGraphRAGRetrievalSettings,
    RetrievalMethodId,
)
from graph_memory.registry.retrieval_builders import build_retrieval_registry
from graph_memory.retrieval.requests import FastGraphRAGRequest
from tests.test_phase1_real_retrieval import FakeEncoder, retrieval_graphs, retrieval_ranking_requests


def test_fast_graphrag_registry_builds_graph_backed_execution_tasks() -> None:
    settings = FastGraphRAGRetrievalSettings(
        top_k=2,
        encoder=DenseEncoderSettings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            query_prefix="query: ",
            passage_prefix="passage: ",
            batch_size=8,
        ),
    )

    built = build_retrieval_registry().build(
        settings,
        FastGraphRAGBuildPayload(
            ranking_requests=retrieval_ranking_requests(),
            graphs=retrieval_graphs(),
            dense_encoder=FakeEncoder(),
        ),
    )

    assert built.method.name == "fast_graphrag"
    assert isinstance(built.execution_tasks[0].method_request, FastGraphRAGRequest)
    assert built.execution_tasks[0].method_request.task_id == built.execution_tasks[0].text_request.task_id
    assert built.execution_tasks[0].method_request.knowledge_graph.entities
    assert built.provenance.encoder == settings.encoder


def test_fast_graphrag_supports_path_metrics() -> None:
    registry = build_method_registry()

    assert registry.supports_path_metrics(RetrievalMethodId.FAST_GRAPHRAG)
