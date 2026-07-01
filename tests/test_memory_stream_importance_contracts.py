from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

import graph_memory.registry.retrieval_builders as retrieval_builders
import graph_memory.retrieval.tuning.memory_stream as memory_stream_tuning
from graph_memory.registry import Registry
from graph_memory.registry.retrieval import (
    DenseEncoderSettings,
    MemoryStreamBuildPayload,
    MemoryStreamRetrievalSettings,
)
from graph_memory.retrieval.methods.flat.dense import DenseConfig
from graph_memory.retrieval.methods.memory_stream.config import MemoryStreamScoringConfig
from graph_memory.retrieval.methods.memory_stream.importance import request_importance_scores
from graph_memory.retrieval.requests import DenseRuntime, PositionRecencySpec, TemporalMemoryRankingRequest, TextCandidate
from tests.test_phase1_real_retrieval import CountingFakeEncoder, FakeEncoder


def _request(
    importance_by_item_id: Mapping[str, float] | None = None,
) -> TemporalMemoryRankingRequest:
    return TemporalMemoryRankingRequest(
        task_id="longmem_q1",
        query_text="Where did I say I planned to meet Alex?",
        candidates=(
            TextCandidate(item_id="m0", text="Meet Alex at the library.", metadata={}),
            TextCandidate(item_id="m1", text="A distractor memory.", metadata={}),
        ),
        importance_by_item_id=importance_by_item_id or {},
        recency=PositionRecencySpec(position_by_item_id={"m0": 0, "m1": 1}),
        metadata={},
    )


def test_request_importance_helper_rejects_non_numeric_scores() -> None:
    request = _request(cast(Mapping[str, float], {"m0": "high", "m1": 0.0}))

    with pytest.raises(ValueError, match="importance must be numeric"):
        request_importance_scores(request, require_complete=True)


def test_memory_stream_builder_calls_shared_request_importance_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_request_importance_scores(
        request: TemporalMemoryRankingRequest,
        *,
        require_complete: bool,
    ) -> dict[str, float]:
        calls.append((request.task_id, require_complete))
        return {"m0": 2.0, "m1": 0.0}

    monkeypatch.setattr(retrieval_builders, "request_importance_scores", fake_request_importance_scores)

    built = Registry.retrieval.build(
        MemoryStreamRetrievalSettings(
            top_k=2,
            encoder=DenseEncoderSettings(model_name="fake-model", query_prefix="query: ", passage_prefix="passage: "),
            scoring=MemoryStreamScoringConfig(importance_weight=1.0),
        ),
        MemoryStreamBuildPayload(
            temporal_requests=[_request()],
            dense_encoder=FakeEncoder(),
        ),
    )

    assert calls == [("longmem_q1", True)]
    method_request = built.execution_tasks[0].method_request
    assert isinstance(method_request, TemporalMemoryRankingRequest)
    assert method_request.importance_by_item_id == {"m0": 2.0, "m1": 0.0}


def test_memory_stream_tuning_calls_shared_request_importance_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_request_importance_scores(
        request: TemporalMemoryRankingRequest,
        *,
        require_complete: bool,
    ) -> dict[str, float]:
        calls.append((request.task_id, require_complete))
        return {"m0": 2.0, "m1": 0.0}

    monkeypatch.setattr(memory_stream_tuning, "request_importance_scores", fake_request_importance_scores)

    _ = memory_stream_tuning.precompute_memory_stream_signal_cache(
        temporal_requests=[_request()],
        importance_artifact=None,
        dense_runtime=DenseRuntime(
            config=DenseConfig(model_name="fake"),
            encoder=CountingFakeEncoder(),
        ),
        require_complete_request_importance=True,
    )

    assert calls == [("longmem_q1", True)]
