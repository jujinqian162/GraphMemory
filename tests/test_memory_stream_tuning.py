from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import pytest

import graph_memory.retrieval.tuning.memory_stream as memory_stream_tuning
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTemporalMemoryRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQALabelRecord, HotpotQARankingRecord
from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.evaluation.suites import longmemeval_metric_suite
from graph_memory.registry.retrieval import RetrievalMethodId
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.memory_stream.artifact import (
    importance_content_digest,
)
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
)
from graph_memory.retrieval.methods.memory_stream.contracts import ImportanceArtifact
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
from graph_memory.retrieval.requests import (
    DenseRuntime,
    TemporalMemoryRankingRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.retrieval.tuning.memory_stream import (
    MemoryStreamSignalCache,
    run_memory_stream_from_signal_cache,
    tune_memory_stream,
)
from graph_memory.retrieval.tuning.memory_stream_grid import (
    memory_stream_grid_from_record,
)
from graph_memory.retrieval.tuning.selection import longmemeval_retrieval_candidate_key
from tests.test_phase1_real_retrieval import (
    CountingFakeEncoder,
    retrieval_graphs,
    retrieval_task_inputs,
    retrieval_task_labels,
)


def _temporal_requests(task_inputs: Sequence[HotpotQARankingRecord]) -> list[TemporalMemoryRankingRequest]:
    projector = HotpotQAToTemporalMemoryRankingRequest()
    return [projector.project(task_input, {}) for task_input in task_inputs]


def _evidence_labels(labels: Sequence[HotpotQALabelRecord]) -> list[EvidenceLabel]:
    return [
        EvidenceLabel(
            task_id=label["task_id"],
            gold_answer=label["gold_answer"],
            gold_evidence_item_ids=tuple(label["gold_evidence_sentence_ids"]),
            gold_dependency_edges=tuple((edge[0], edge[1]) for edge in label["gold_dependency_edges"]),
        )
        for label in labels
    ]


def _importance_artifact(
    temporal_requests: list[TemporalMemoryRankingRequest],
) -> ImportanceArtifact:
    return {
        "schema_version": 1,
        "method": "memory_stream",
        "tasks": [
            {
                "task_id": request.task_id,
                "content_digest": importance_content_digest(request),
                "scores": {
                    candidate.item_id: 10 if candidate.item_id == "m1" else 1
                    for candidate in request.candidates
                },
            }
            for request in temporal_requests
        ],
    }


def test_memory_stream_grid_allows_single_value_fields() -> None:
    grid = memory_stream_grid_from_record(
        {
            "relevance_weight": [1.0],
            "recency_weight": [0.0],
            "importance_weight": [0.1],
            "recency_decay": [0.99],
        }
    )

    assert grid == [
        MemoryStreamScoringConfig(
            relevance_weight=1.0,
            recency_weight=0.0,
            importance_weight=0.1,
            recency_decay=0.99,
        )
    ]


def test_memory_stream_grid_expands_full_product_without_deduplication() -> None:
    grid = memory_stream_grid_from_record(
        {
            "relevance_weight": [1.0, 1.0],
            "recency_weight": [0.0],
            "importance_weight": [0.1, 0.2],
            "recency_decay": [0.99],
        }
    )

    assert [
        (config.relevance_weight, config.importance_weight)
        for config in grid
    ] == [
        (1.0, 0.1),
        (1.0, 0.2),
        (1.0, 0.1),
        (1.0, 0.2),
    ]


@pytest.mark.parametrize(
    "record, match",
    [
        (
            {
                "relevance_weight": [1.0],
                "recency_weight": [0.0],
                "importance_weight": [0.1],
            },
            "missing fields",
        ),
        (
            {
                "relevance_weight": [1.0],
                "recency_weight": [0.0],
                "importance_weight": [0.1],
                "recency_decay": [0.99],
                "unknown": [1.0],
            },
            "unsupported fields",
        ),
    ],
)


def test_memory_stream_grid_rejects_invalid_fields(
    record: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = memory_stream_grid_from_record(record)


def test_memory_stream_tuning_computes_dense_seed_once_for_all_candidates() -> None:
    encoder = CountingFakeEncoder()
    temporal_requests = _temporal_requests(retrieval_task_inputs())
    grid = [
        MemoryStreamScoringConfig(importance_weight=0.0),
        MemoryStreamScoringConfig(importance_weight=0.1),
        MemoryStreamScoringConfig(importance_weight=1.0),
    ]

    _, candidate_rows = tune_memory_stream(
        temporal_requests=temporal_requests,
        labels=_evidence_labels(retrieval_task_labels()),
        graphs=retrieval_graphs(),
        importance_artifact=_importance_artifact(temporal_requests),
        grid=grid,
        top_k=2,
        dense_runtime=DenseRuntime(
            config=DenseConfig(model_name="fake"),
            encoder=encoder,
        ),
    )

    assert encoder.encode_calls == 1
    assert [row["config"] for row in candidate_rows] == [
        {
            "relevance_weight": config.relevance_weight,
            "recency_weight": config.recency_weight,
            "importance_weight": config.importance_weight,
            "recency_decay": config.recency_decay,
        }
        for config in grid
    ]



def test_memory_stream_tuning_uses_request_importance_without_artifact() -> None:
    encoder = CountingFakeEncoder()
    temporal_requests = _temporal_requests(retrieval_task_inputs())
    grid = [
        MemoryStreamScoringConfig(
            relevance_weight=1.0,
            recency_weight=0.1,
            importance_weight=0.0,
            recency_decay=0.99,
        )
    ]

    selected, candidate_rows = tune_memory_stream(
        temporal_requests=temporal_requests,
        labels=_evidence_labels(retrieval_task_labels()),
        graphs=retrieval_graphs(),
        importance_artifact=None,
        grid=grid,
        top_k=2,
        dense_runtime=DenseRuntime(
            config=DenseConfig(model_name="fake"),
            encoder=encoder,
        ),
    )

    assert encoder.encode_calls == 1
    assert selected["importance_weight"] == 0.0
    assert candidate_rows[0]["config"]["importance_weight"] == 0.0


def test_memory_stream_tuning_can_use_longmemeval_metric_suite() -> None:
    temporal_requests = [
        TemporalMemoryRankingRequest(
            task_id="longmem_q1",
            query_text="Where did I say I would visit in Paris?",
            candidates=(
                TextCandidate(
                    item_id="m0",
                    text="I will visit Paris.",
                    metadata={"position": 0, "session_id": "s1"},
                ),
                TextCandidate(
                    item_id="m1",
                    text="I bought groceries.",
                    metadata={"position": 1, "session_id": "s2"},
                ),
            ),
            importance_by_item_id={},
            metadata={"position_by_item_id": {"m0": 0, "m1": 1}},
        )
    ]
    graphs: list[MemoryGraph] = [
        {
            "task_id": "longmem_q1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "Where?"},
                {
                    "id": "m0",
                    "node_type": "graph_item",
                    "node_kind": "conversation_turn",
                    "text": "I will visit Paris.",
                    "metadata": {"session_id": "s1"},
                },
                {
                    "id": "m1",
                    "node_type": "graph_item",
                    "node_kind": "conversation_turn",
                    "text": "I bought groceries.",
                    "metadata": {"session_id": "s2"},
                },
            ],
            "edges": [],
        }
    ]

    selected, candidate_rows = tune_memory_stream(
        temporal_requests=temporal_requests,
        labels=[
            EvidenceLabel(
                task_id="longmem_q1",
                gold_answer="Paris.",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(),
                gold_session_ids=("s1",),
            )
        ],
        graphs=graphs,
        importance_artifact=None,
        grid=[
            MemoryStreamScoringConfig(
                relevance_weight=1.0,
                recency_weight=0.0,
                importance_weight=0.0,
                recency_decay=0.99,
            )
        ],
        top_k=2,
        dense_runtime=DenseRuntime(
            config=DenseConfig(model_name="fake"),
            encoder=CountingFakeEncoder(),
        ),
        metric_suite=longmemeval_metric_suite(),
        selection_key=longmemeval_retrieval_candidate_key,
    )

    assert selected["importance_weight"] == 0.0
    assert candidate_rows[0].get("Turn Recall@5") == 1.0
    assert candidate_rows[0].get("Session Recall@5") == 1.0
    assert "Recall@5" not in candidate_rows[0]
    assert "Connected Evidence Recall@10" not in candidate_rows[0]


def test_memory_stream_tuning_uses_retrieval_selection_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporal_requests = _temporal_requests(retrieval_task_inputs())
    grid = [
        MemoryStreamScoringConfig(importance_weight=0.1),
        MemoryStreamScoringConfig(importance_weight=0.9),
    ]
    monkeypatch.setattr(
        memory_stream_tuning,
        "retrieval_candidate_key",
        lambda row: (float(row["config"]["importance_weight"]),),
    )

    selected, _ = tune_memory_stream(
        temporal_requests=temporal_requests,
        labels=_evidence_labels(retrieval_task_labels()),
        graphs=retrieval_graphs(),
        importance_artifact=_importance_artifact(temporal_requests),
        grid=grid,
        top_k=2,
        dense_runtime=DenseRuntime(
            config=DenseConfig(model_name="fake"),
            encoder=CountingFakeEncoder(),
        ),
    )

    assert selected["importance_weight"] == 0.9


@dataclass(frozen=True)
class _FakeDenseRanker:
    method_name: str = RetrievalMethodId.DENSE.value

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        assert request.task_id == "hotpot_ex1"
        return [
            RankedNode(node_id="m0", score=3.0),
            RankedNode(node_id="m1", score=2.0),
            RankedNode(node_id="m2", score=1.0),
        ]


def test_cached_memory_stream_execution_matches_formal_method_ranking() -> None:
    task_inputs = retrieval_task_inputs()
    task_input = task_inputs[0]
    temporal_requests = _temporal_requests(task_inputs)
    artifact = _importance_artifact(temporal_requests)
    scoring = MemoryStreamScoringConfig(
        relevance_weight=1.0,
        recency_weight=0.5,
        importance_weight=0.8,
        recency_decay=0.9,
    )
    cache = MemoryStreamSignalCache(
        relevance_by_task_id={
            task_input["task_id"]: {"m0": 1.0, "m1": 0.5, "m2": 0.0}
        },
        importance_by_task_id={
            task_input["task_id"]: {"m0": 0.0, "m1": 1.0, "m2": 0.0}
        },
        seed_latency_ms_by_task_id={task_input["task_id"]: 0.0},
    )

    cached = run_memory_stream_from_signal_cache(
        temporal_requests=temporal_requests,
        signal_cache=cache,
        top_k=2,
        scoring=scoring,
    )
    method = MemoryStreamMethod(
        name=RetrievalMethodId.MEMORY_STREAM.value,
        dense_seed_ranker=_FakeDenseRanker(),
        importance_by_task_id={
            record["task_id"]: record for record in artifact["tasks"]
        },
        scoring=scoring,
    )
    formal_request = HotpotQAToTemporalMemoryRankingRequest().project(
        task_input,
        method.importance_scores_for_task(task_input["task_id"]),
    )
    formal = method.rank_task(formal_request, top_k=2)

    assert cached[0]["ranked_nodes"] == [
        {"node_id": node.node_id, "score": node.score}
        for node in formal.ranked_nodes
    ]
