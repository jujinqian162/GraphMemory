from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.retrieval.contracts import RankedNode, RetrievalMethodResult
from graph_memory.retrieval.methods.memory_stream.artifact import importance_content_digest
from graph_memory.retrieval.methods.memory_stream.contracts import TaskImportanceRecord
from graph_memory.retrieval.methods.memory_stream.method import MemoryStreamMethod
from graph_memory.retrieval.methods.memory_stream.scoring import (
    MemoryStreamWeights,
    NormalizedMemoryStreamSignals,
    RawMemoryStreamSignals,
    combine_memory_stream_signals,
    normalize_memory_stream_signals,
    normalize_task_signal,
    rank_memory_stream_scores,
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


def test_combine_memory_stream_signals_applies_weights() -> None:
    normalized = NormalizedMemoryStreamSignals(
        relevance_by_node_id={"m0": 1.0, "m1": 0.0},
        recency_by_node_id={"m0": 0.0, "m1": 1.0},
        importance_by_node_id={"m0": 0.0, "m1": 1.0},
    )

    combined = combine_memory_stream_signals(
        normalized,
        weights=MemoryStreamWeights(relevance=2.0, recency=3.0, importance=4.0),
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
    method = MemoryStreamMethod(
        name="memory_stream",
        dense_seed_ranker=_FakeDenseRanker(),
        importance_by_task_id={task_input["task_id"]: _importance_record(task_input)},
        weights=MemoryStreamWeights(relevance=1.0, recency=1.0, importance=1.0),
        recency_decay=1.0,
    )

    result = method.rank_task(task_input, top_k=2)

    assert result == RetrievalMethodResult(
        ranked_nodes=[
            RankedNode(node_id="m1", score=1.0),
            RankedNode(node_id="m2", score=1.0),
            RankedNode(node_id="m0", score=0.0),
        ]
    )


def test_memory_stream_settings_default_to_dense_with_weak_importance_and_no_recency() -> None:
    settings = MemoryStreamRetrievalSettings(
        top_k=3,
        encoder=DenseEncoderSettings(model_name="fake", query_prefix="query: ", passage_prefix="passage: "),
    )

    assert settings.relevance_weight == 1.0
    assert settings.recency_weight == 0.0
    assert settings.importance_weight == 0.01


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "relevance_weight": -1.0,
            },
            "relevance_weight",
        ),
        (
            {
                "relevance_weight": 0.0,
                "recency_weight": 0.0,
                "importance_weight": 0.0,
            },
            "at least one",
        ),
        (
            {
                "recency_decay": 0.0,
            },
            "recency_decay",
        ),
        (
            {
                "recency_decay": 1.1,
            },
            "recency_decay",
        ),
    ],
)
def test_memory_stream_settings_validate_weights_and_decay(kwargs: dict[str, float], match: str) -> None:
    base = {
        "top_k": 3,
        "encoder": DenseEncoderSettings(model_name="fake", query_prefix="query: ", passage_prefix="passage: "),
    }

    with pytest.raises(ValueError, match=match):
        _ = MemoryStreamRetrievalSettings(**base, **kwargs)  # pyright: ignore[reportArgumentType]
