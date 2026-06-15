from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.config import (
    MemoryStreamScoringConfig,
    memory_stream_scoring_config_record,
    parse_memory_stream_scoring_config,
)
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
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


def _task_input() -> MemoryTaskInput:
    return {
        "task_id": "hotpot_ms_1",
        "query": "Which river runs through Paris?",
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
                "text": "Paris is a major city in France.",
                "source": "Paris",
                "sentence_id": 1,
                "position": 2,
            },
        ],
    }


def _importance_record(task_input: MemoryTaskInput) -> TaskImportanceRecord:
    return {
        "task_id": task_input["task_id"],
        "content_digest": importance_content_digest(task_input),
        "scores": {"m0": 1, "m1": 10, "m2": 1},
    }


@dataclass(frozen=True)
class _FakeDenseRanker:
    method_name: str = "dense"

    def rank(self, task_input: MemoryTaskInput) -> list[RankedNode]:
        assert task_input["task_id"] == "hotpot_ms_1"
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

    result = method.rank_task(task_input, top_k=2)
    normalized = normalize_memory_stream_signals(
        RawMemoryStreamSignals(
            relevance_by_node_id={"m2": 9.0, "m0": 5.0, "m1": 5.0},
            recency_by_node_id=pseudo_recency_scores(task_input, decay=1.0),
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
