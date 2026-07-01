from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph_memory.contracts.errors import ContractValidationError
from graph_memory.datasets.hotpotqa.projectors import HotpotQAToTemporalMemoryRankingRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    memory_stream_scoring_config_record,
    parse_memory_stream_scoring_config,
)
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
import graph_memory.retrieval.methods.memory_stream.scoring as memory_stream_scoring
from graph_memory.retrieval.methods.memory_stream.scoring import (
    NormalizedMemoryStreamSignals,
    RawMemoryStreamSignals,
    normalize_memory_stream_signals,
    normalize_task_signal,
    pseudo_recency_scores,
    rank_memory_stream_scores,
    score_memory_stream,
)
from graph_memory.registry.retrieval import DenseEncoderSettings, MemoryStreamRetrievalSettings
from graph_memory.retrieval.requests import RealTimeRecencySpec, TemporalMemoryRankingRequest, TextCandidate, TextRankingRequest


def _task_input() -> HotpotQARankingRecord:
    return {
        "task_id": "hotpot_ms_1",
        "question": "Which river runs through Paris?",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "text": "The Eiffel Tower is in Paris.",
                "title": "Eiffel Tower",
                "sentence_index": 0,
                "position": 0,
            },
            {
                "sentence_id": "m1",
                "text": "The Seine runs through Paris.",
                "title": "Paris",
                "sentence_index": 0,
                "position": 1,
            },
            {
                "sentence_id": "m2",
                "text": "Paris is a major city in France.",
                "title": "Paris",
                "sentence_index": 1,
                "position": 2,
            },
        ],
    }


def _temporal_request(task_input: HotpotQARankingRecord):
    return HotpotQAToTemporalMemoryRankingRequest().project(task_input, {})


def _importance_record(task_input: HotpotQARankingRecord) -> TaskImportanceRecord:
    return {
        "task_id": task_input["task_id"],
        "content_digest": importance_content_digest(_temporal_request(task_input)),
        "scores": {"m0": 1, "m1": 10, "m2": 1},
    }


@dataclass(frozen=True)
class _FakeDenseRanker:
    method_name: str = "dense"

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        assert request.task_id == "hotpot_ms_1"
        return [
            RankedNode(node_id="m2", score=9.0),
            RankedNode(node_id="m0", score=5.0),
            RankedNode(node_id="m1", score=5.0),
        ]


def test_normalize_task_signal_maps_constant_scores_to_zero() -> None:
    assert normalize_task_signal({"m0": 7.0, "m1": 7.0}) == {"m0": 0.0, "m1": 0.0}


def test_normalize_memory_stream_signals_uses_all_nodes_from_the_task() -> None:
    normalized = normalize_memory_stream_signals(
        RawMemoryStreamSignals(
            relevance_by_node_id={"m0": 3.0},
            recency_by_node_id={"m1": 4.0},
            importance_by_node_id={"m0": 1.0, "m1": 5.0},
        )
    )

    assert set(normalized.relevance_by_node_id) == {"m0", "m1"}
    assert set(normalized.recency_by_node_id) == {"m0", "m1"}
    assert set(normalized.importance_by_node_id) == {"m0", "m1"}
    assert normalized.relevance_by_node_id["m1"] == 0.0
    assert normalized.recency_by_node_id["m0"] == 0.0


def test_score_memory_stream_applies_scoring_config() -> None:
    normalized = NormalizedMemoryStreamSignals(
        relevance_by_node_id={"m0": 1.0, "m1": 0.0},
        recency_by_node_id={"m0": 0.0, "m1": 1.0},
        importance_by_node_id={"m0": 0.0, "m1": 1.0},
    )

    combined = score_memory_stream(
        normalized,
        config=MemoryStreamScoringConfig(
            relevance_weight=2.0,
            recency_weight=3.0,
            importance_weight=4.0,
        ),
    )

    assert combined == {"m0": 2.0, "m1": 7.0}


def test_rank_memory_stream_scores_breaks_ties_by_node_id() -> None:
    assert rank_memory_stream_scores({"m1": 2.0, "m0": 2.0, "m2": 1.0}) == [
        ("m0", 2.0),
        ("m1", 2.0),
        ("m2", 1.0),
    ]


def test_memory_stream_real_time_recency_uses_typed_request_data_without_metadata() -> None:
    request = TemporalMemoryRankingRequest(
        task_id="longmem_q1",
        query_text="What did I discuss recently?",
        candidates=(
            TextCandidate(item_id="m_old", text="Older memory.", metadata={}),
            TextCandidate(item_id="m_recent", text="Recent memory.", metadata={}),
        ),
        importance_by_item_id={},
        recency=RealTimeRecencySpec(
            question_datetime="2024-01-10T12:00:00",
            datetime_by_item_id={
                "m_old": "2024-01-08T12:00:00",
                "m_recent": "2024-01-09T12:00:00",
            },
        ),
        metadata={
            "recency_mode": "position",
            "position_by_item_id": {"m_old": 0, "m_recent": 1},
        },
    )

    scores = memory_stream_scoring.memory_stream_recency_scores(request, decay=0.5)

    assert scores == {
        "m_old": pytest.approx(0.25),
        "m_recent": pytest.approx(0.5),
    }


def test_memory_stream_real_time_recency_uses_candidate_datetimes_not_positions() -> None:
    request = TemporalMemoryRankingRequest(
        task_id="longmem_q1",
        query_text="What did I discuss recently?",
        candidates=(
            TextCandidate(item_id="m_old", text="Older memory.", metadata={}),
            TextCandidate(item_id="m_recent", text="Recent memory.", metadata={}),
        ),
        importance_by_item_id={},
        recency=RealTimeRecencySpec(
            question_datetime="2024-01-10T12:00:00",
            datetime_by_item_id={
                "m_old": "2024-01-08T12:00:00",
                "m_recent": "2024-01-09T12:00:00",
            },
        ),
        metadata={
            "recency_mode": "real_time",
            "question_datetime": "2024-01-10T12:00:00",
            "datetime_by_item_id": {
                "m_old": "2024-01-08T12:00:00",
                "m_recent": "2024-01-09T12:00:00",
            },
            "position_by_item_id": {"m_old": 1, "m_recent": 0},
        },
    )

    scores = memory_stream_scoring.memory_stream_recency_scores(request, decay=0.5)

    assert scores == {
        "m_old": pytest.approx(0.25),
        "m_recent": pytest.approx(0.5),
    }


def test_memory_stream_real_time_recency_accepts_longmemeval_raw_datetime_format() -> None:
    request = TemporalMemoryRankingRequest(
        task_id="longmem_q1",
        query_text="What did I discuss yesterday?",
        candidates=(TextCandidate(item_id="m0", text="Memory.", metadata={}),),
        importance_by_item_id={},
        recency=RealTimeRecencySpec(
            question_datetime="2023/05/30 (Tue) 23:40",
            datetime_by_item_id={"m0": "2023/05/29 (Mon) 23:40"},
        ),
        metadata={
            "recency_mode": "real_time",
            "question_datetime": "2023/05/30 (Tue) 23:40",
            "datetime_by_item_id": {"m0": "2023/05/29 (Mon) 23:40"},
        },
    )

    scores = memory_stream_scoring.memory_stream_recency_scores(request, decay=0.5)

    assert scores == {"m0": pytest.approx(0.5)}


def test_memory_stream_real_time_recency_anchors_to_latest_visible_datetime() -> None:
    request = TemporalMemoryRankingRequest(
        task_id="longmem_future",
        query_text="What happened?",
        candidates=(
            TextCandidate(item_id="m_future", text="Future relative to question.", metadata={}),
            TextCandidate(item_id="m_latest", text="Latest visible memory.", metadata={}),
        ),
        importance_by_item_id={},
        recency=RealTimeRecencySpec(
            question_datetime="2024-01-10T12:00:00",
            datetime_by_item_id={
                "m_future": "2024-01-11T12:00:00",
                "m_latest": "2024-01-12T12:00:00",
            },
        ),
        metadata={
            "recency_mode": "real_time",
            "question_datetime": "2024-01-10T12:00:00",
            "datetime_by_item_id": {
                "m_future": "2024-01-11T12:00:00",
                "m_latest": "2024-01-12T12:00:00",
            },
        },
    )

    scores = memory_stream_scoring.memory_stream_recency_scores(request, decay=0.5)

    assert scores == {
        "m_future": pytest.approx(0.5),
        "m_latest": pytest.approx(1.0),
    }


def test_memory_stream_real_time_recency_requires_question_datetime() -> None:
    request = TemporalMemoryRankingRequest(
        task_id="longmem_q1",
        query_text="What did I discuss recently?",
        candidates=(TextCandidate(item_id="m0", text="Memory.", metadata={}),),
        importance_by_item_id={},
        recency=RealTimeRecencySpec(
            question_datetime="",
            datetime_by_item_id={"m0": "2024-01-09T12:00:00"},
        ),
        metadata={
            "recency_mode": "real_time",
            "datetime_by_item_id": {"m0": "2024-01-09T12:00:00"},
        },
    )

    with pytest.raises(ContractValidationError, match="question_datetime"):
        _ = memory_stream_scoring.memory_stream_recency_scores(request, decay=0.99)


def test_memory_stream_method_ranks_complete_nodes_and_uses_fake_dense_seed_ranker() -> None:
    task_input = _task_input()
    scoring = MemoryStreamScoringConfig(
        relevance_weight=1.0,
        recency_weight=1.0,
        importance_weight=1.0,
        recency_decay=1.0,
    )
    method = MemoryStreamMethod(
        name="memory_stream",
        dense_seed_ranker=_FakeDenseRanker(),
        importance_by_task_id={task_input["task_id"]: _importance_record(task_input)},
        scoring=scoring,
    )
    request = HotpotQAToTemporalMemoryRankingRequest().project(
        task_input,
        method.importance_scores_for_task(task_input["task_id"]),
    )

    result = method.rank_task(request, top_k=2)
    normalized = normalize_memory_stream_signals(
        RawMemoryStreamSignals(
            relevance_by_node_id={"m2": 9.0, "m0": 5.0, "m1": 5.0},
            recency_by_node_id=pseudo_recency_scores(request, decay=1.0),
            importance_by_node_id={"m0": 1.0, "m1": 10.0, "m2": 1.0},
        )
    )
    expected = [
        RankedNode(node_id=node_id, score=score)
        for node_id, score in rank_memory_stream_scores(
            score_memory_stream(normalized, config=scoring)
        )
    ]

    assert result == RetrievalMethodResult(ranked_nodes=expected)


def test_memory_stream_settings_default_to_dense_with_weak_importance_and_no_recency() -> None:
    settings = MemoryStreamRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings(model_name="fake", query_prefix="query: ", passage_prefix="passage: "),
    )

    assert settings.scoring == MemoryStreamScoringConfig()


def test_memory_stream_scoring_config_parser_and_serializer_round_trip() -> None:
    record = {
        "relevance_weight": 1,
        "recency_weight": 0.2,
        "importance_weight": 0.5,
        "recency_decay": 0.99,
    }

    config = parse_memory_stream_scoring_config(record)

    assert config == MemoryStreamScoringConfig(
        relevance_weight=1.0,
        recency_weight=0.2,
        importance_weight=0.5,
        recency_decay=0.99,
    )
    assert memory_stream_scoring_config_record(config) == {
        "relevance_weight": 1.0,
        "recency_weight": 0.2,
        "importance_weight": 0.5,
        "recency_decay": 0.99,
    }


@pytest.mark.parametrize(
    "record, match",
    [
        (
            {
                "relevance_weight": -1.0,
                "recency_weight": 0.0,
                "importance_weight": 0.01,
                "recency_decay": 0.99,
            },
            "relevance_weight",
        ),
        (
            {
                "relevance_weight": 0.0,
                "recency_weight": 0.0,
                "importance_weight": 0.0,
                "recency_decay": 0.99,
            },
            "at least one",
        ),
        (
            {
                "relevance_weight": 1.0,
                "recency_weight": 0.0,
                "importance_weight": 0.01,
                "recency_decay": 0.0,
            },
            "recency_decay",
        ),
        (
            {
                "relevance_weight": 1.0,
                "recency_weight": 0.0,
                "importance_weight": 0.01,
                "recency_decay": 1.1,
            },
            "recency_decay",
        ),
        (
            {
                "relevance_weight": float("inf"),
                "recency_weight": 0.0,
                "importance_weight": 0.01,
                "recency_decay": 0.99,
            },
            "relevance_weight",
        ),
        (
            {
                "relevance_weight": 1.0,
                "recency_weight": 0.0,
                "importance_weight": 0.01,
                "recency_decay": 0.99,
                "unknown": 1.0,
            },
            "unsupported fields",
        ),
        (
            {
                "relevance_weight": 1.0,
                "recency_weight": 0.0,
                "importance_weight": 0.01,
            },
            "missing fields",
        ),
    ],
)


def test_memory_stream_scoring_config_parser_validates_record(
    record: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = parse_memory_stream_scoring_config(record)
