from __future__ import annotations

import pytest

from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    FlatRetrievalBuildPayload,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    RetrievalMethodId,
    SeedRetrievalSettings,
    require_payload,
)
from graph_memory.registry.retrieval_builders import RETRIEVAL_REGISTRY
from tests.test_phase2_rgcn_training import tiny_graphs, tiny_task_inputs


def test_require_payload_accepts_expected_payload_type() -> None:
    payload = FlatRetrievalBuildPayload(task_inputs=tiny_task_inputs())

    assert require_payload(payload, FlatRetrievalBuildPayload, method="bm25") is payload


def test_require_payload_rejects_incompatible_payload_type() -> None:
    with pytest.raises(TypeError, match="bm25 expected FlatRetrievalBuildPayload"):
        require_payload(object(), FlatRetrievalBuildPayload, method="bm25")


def test_bm25_builder_accepts_flat_payload() -> None:
    method = RETRIEVAL_REGISTRY.build(
        Bm25RetrievalSettings(top_k=2),
        FlatRetrievalBuildPayload(task_inputs=tiny_task_inputs()),
    )

    assert method.name == "bm25"


def test_graph_rerank_builder_requires_graph_payload() -> None:
    settings = GraphRerankRetrievalSettings(
        method=RetrievalMethodId.BM25_GRAPH_RERANK,
        top_k=2,
        seed=SeedRetrievalSettings(method=RetrievalMethodId.BM25),
        rerank=GraphRerankSettings(),
    )

    with pytest.raises(TypeError, match="bm25_graph_rerank expected GraphRerankBuildPayload"):
        RETRIEVAL_REGISTRY.build(settings, FlatRetrievalBuildPayload(task_inputs=tiny_task_inputs()))


def test_graph_rerank_builder_accepts_graph_payload() -> None:
    settings = GraphRerankRetrievalSettings(
        method=RetrievalMethodId.BM25_GRAPH_RERANK,
        top_k=2,
        seed=SeedRetrievalSettings(method=RetrievalMethodId.BM25),
        rerank=GraphRerankSettings(),
    )

    method = RETRIEVAL_REGISTRY.build(
        settings,
        GraphRerankBuildPayload(
            task_inputs=tiny_task_inputs(),
            graphs=tiny_graphs(),
            graph_config=None,
        ),
    )

    assert method.name == "bm25_graph_rerank"
