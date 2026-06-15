from __future__ import annotations

from pathlib import Path
from typing import cast

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    Bm25RetrievalSettings,
    DenseEncoderSettings,
    DenseRetrievalSettings,
    FlatRetrievalBuildPayload,
    ImportanceArtifactProvenance,
    GraphRerankBuildPayload,
    GraphRerankRetrievalSettings,
    GraphRerankSettings,
    MemoryStreamBuildPayload,
    MemoryStreamRetrievalSettings,
    RetrievalMethodId,
    SeedRetrievalSettings,
)
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
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


def _memory_stream_task(task_id: str, query: str, memory_items: list[dict[str, object]]) -> MemoryTaskInput:
    return cast(
        MemoryTaskInput,
        cast(
            object,
            {
                "task_id": task_id,
                "query": query,
                "memory_items": memory_items,
            },
        ),
    )


def test_memory_stream_builder_selects_current_task_importance_and_records_provenance(tmp_path: Path) -> None:
    task_inputs: list[MemoryTaskInput] = [
        _memory_stream_task(
            "hotpot_ms_1",
            "Which river runs through Paris?",
            [
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
            ],
        )
    ]
    extra_task = _memory_stream_task(
        "hotpot_ms_extra",
        "Which city has the Louvre?",
        [
            {
                "id": "x0",
                "node_type": "document_sentence",
                "text": "The Louvre is in Paris.",
                "source": "Louvre",
                "sentence_id": 0,
                "position": 0,
            }
        ],
    )
    artifact: ImportanceArtifact = {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": [
            {
                "task_id": task["task_id"],
                "content_digest": importance_content_digest(task),
                "scores": {item["id"]: index + 1 for index, item in enumerate(task["memory_items"])},
            }
            for task in (*task_inputs, extra_task)
        ],
    }

    built = Registry.retrieval.build(
        MemoryStreamRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings(model_name="fake-model", query_prefix="query: ", passage_prefix="passage: "),
            recency_decay=1.0,
        ),
        MemoryStreamBuildPayload(
            task_inputs=task_inputs,
            importance_artifact=artifact,
            importance_path=tmp_path / "dev.first_1000.importance.json",
            importance_sha256="abc123",
            dense_encoder=FakeEncoder(),
        ),
    )

    assert isinstance(built.method, MemoryStreamMethod)
    assert built.method.name == "memory_stream"
    assert built.method.dense_seed_ranker.method_name == "dense"
    assert set(built.method.importance_by_task_id) == {"hotpot_ms_1"}
    assert built.provenance.importance == ImportanceArtifactProvenance(
        path=tmp_path / "dev.first_1000.importance.json",
        sha256="abc123",
        schema_version=1,
    )
