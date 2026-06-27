from __future__ import annotations

from pathlib import Path
from typing import cast

from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTemporalMemoryRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
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
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
from tests.test_phase1_real_retrieval import FakeEncoder, retrieval_graphs, retrieval_ranking_requests


def _temporal_requests(task_inputs: list[HotpotQARankingRecord]):
    projector = HotpotQAToTemporalMemoryRankingRequest()
    return [projector.project(task_input, {}) for task_input in task_inputs]


def test_registry_builds_bm25_method_from_settings_without_dense_fields() -> None:
    settings = Bm25RetrievalSettings(top_k=2)

    built = Registry.retrieval.build(settings, FlatRetrievalBuildPayload(ranking_requests=retrieval_ranking_requests()))
    predictions = run_retrieval(retrieval_method=built.method, tasks=built.execution_tasks, top_k=settings.top_k)

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
        FlatRetrievalBuildPayload(ranking_requests=retrieval_ranking_requests(), dense_encoder=FakeEncoder()),
    )
    predictions = run_retrieval(retrieval_method=built.method, tasks=built.execution_tasks, top_k=settings.top_k)

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
        GraphRerankBuildPayload(ranking_requests=retrieval_ranking_requests(), graphs=retrieval_graphs()),
    )
    predictions = run_retrieval(retrieval_method=built.method, tasks=built.execution_tasks, top_k=settings.top_k)

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


def _as_int(value: object) -> int:
    return int(cast(int | str, value))


def _memory_stream_task(task_id: str, query: str, memory_items: list[dict[str, object]]) -> HotpotQARankingRecord:
    return cast(
        HotpotQARankingRecord,
        cast(
            object,
            {
                "task_id": task_id,
                "question": query,
                "candidate_sentences": [
                    {
                        "sentence_id": str(item["id"]),
                        "title": str(item["source"]),
                        "sentence_index": _as_int(item["sentence_id"]),
                        "position": _as_int(item["position"]),
                        "text": str(item["text"]),
                    }
                    for item in memory_items
                ],
            },
        ),
    )


def test_memory_stream_builder_selects_current_task_importance_and_records_provenance(tmp_path: Path) -> None:
    task_inputs: list[HotpotQARankingRecord] = [
        _memory_stream_task(
            "hotpot_ms_1",
            "Which river runs through Paris?",
            [
                {
                    "id": "m0",
                    "text": "The Eiffel Tower is in Paris.",
                    "source": "Eiffel Tower",
                    "sentence_id": 0,
                    "position": 0,
                },
                {
                    "id": "m1",
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
                "text": "The Louvre is in Paris.",
                "source": "Louvre",
                "sentence_id": 0,
                "position": 0,
            }
        ],
    )
    temporal_requests = _temporal_requests([*task_inputs, extra_task])
    selected_temporal_requests = temporal_requests[: len(task_inputs)]
    artifact: ImportanceArtifact = {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": [
            {
                "task_id": request.task_id,
                "content_digest": importance_content_digest(request),
                "scores": {candidate.item_id: index + 1 for index, candidate in enumerate(request.candidates)},
            }
            for request in temporal_requests
        ],
    }
    built = Registry.retrieval.build(
        MemoryStreamRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings(model_name="fake-model", query_prefix="query: ", passage_prefix="passage: "),
            scoring=MemoryStreamScoringConfig(recency_decay=1.0),
        ),
        MemoryStreamBuildPayload(
            temporal_requests=selected_temporal_requests,
            importance_artifact=artifact,
            importance_path=tmp_path / "dev.first_1000.importance.json",
            importance_sha256="abc123",
            dense_encoder=FakeEncoder(),
            scoring_config=MemoryStreamScoringConfig(
                relevance_weight=0.0,
                importance_weight=1.0,
            ),
        ),
    )

    assert isinstance(built.method, MemoryStreamMethod)
    assert built.method.name == "memory_stream"
    assert built.method.scoring == MemoryStreamScoringConfig(
        relevance_weight=0.0,
        importance_weight=1.0,
    )
    assert built.method.dense_seed_ranker.method_name == "dense"
    assert set(built.method.importance_by_task_id) == {"hotpot_ms_1"}
    assert built.provenance.importance == ImportanceArtifactProvenance(
        path=tmp_path / "dev.first_1000.importance.json",
        sha256="abc123",
        schema_version=1,
    )


def test_memory_stream_builder_uses_request_importance_without_external_artifact() -> None:
    from graph_memory.retrieval.requests import TemporalMemoryRankingRequest, TextCandidate

    request = TemporalMemoryRankingRequest(
        task_id="longmem_q1",
        query_text="Where did I say I planned to meet Alex?",
        candidates=(
            TextCandidate(item_id="m0", text="Meet Alex at the library.", metadata={"position": 0}),
            TextCandidate(item_id="m1", text="A distractor memory.", metadata={"position": 1}),
        ),
        importance_by_item_id={"m0": 0.0, "m1": 0.0},
        metadata={"position_by_item_id": {"m0": 0, "m1": 1}},
    )

    built = Registry.retrieval.build(
        MemoryStreamRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings(model_name="fake-model", query_prefix="query: ", passage_prefix="passage: "),
            scoring=MemoryStreamScoringConfig(
                relevance_weight=1.0,
                recency_weight=0.1,
                importance_weight=0.0,
            ),
        ),
        MemoryStreamBuildPayload(
            temporal_requests=[request],
            dense_encoder=FakeEncoder(),
        ),
    )

    assert isinstance(built.method, MemoryStreamMethod)
    assert built.provenance.importance is None
    method_request = built.execution_tasks[0].method_request
    assert isinstance(method_request, TemporalMemoryRankingRequest)
    assert method_request.importance_by_item_id == {"m0": 0.0, "m1": 0.0}
