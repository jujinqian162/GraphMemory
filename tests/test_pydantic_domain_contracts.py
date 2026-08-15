from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonNegativeInt,
    StrictBool,
)
from graph_memory.retrieval.contracts import RetrievalMethodResult
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import (
    RankingMethodRequest,
    TextCandidate,
    TextRankingRequest,
)


class _ScientificProbe(DomainModel):
    count: NonNegativeInt
    enabled: StrictBool
    score: FiniteFloat
    item_ids: tuple[str, ...]


def test_domain_model_is_closed_frozen_strict_and_json_round_trippable() -> None:
    probe = _ScientificProbe(
        count=1,
        enabled=True,
        score=0.5,
        item_ids=("a", "b"),
    )

    assert _ScientificProbe.model_validate_json(probe.model_dump_json()) == probe
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ScientificProbe.model_validate(
            {
                "count": 1,
                "enabled": True,
                "score": 0.5,
                "item_ids": ("a",),
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        _ScientificProbe.model_validate(
            {"count": True, "enabled": 1, "score": math.nan, "item_ids": ("a",)}
        )
    with pytest.raises(ValidationError, match="frozen"):
        setattr(probe, "count", 2)


class _InvalidFirstMethod:
    name: str = RetrievalMethodId.BM25.value

    def __init__(self) -> None:
        self.calls = 0

    def rank_task(
        self, request: RankingMethodRequest, *, top_k: int
    ) -> RetrievalMethodResult:
        _ = request, top_k
        self.calls += 1
        return RetrievalMethodResult(ranked_nodes=())


def test_invalid_first_result_stops_retrieval_before_second_task() -> None:
    requests = tuple(
        TextRankingRequest(
            task_id=f"task-{index}",
            query_text="query",
            candidates=(TextCandidate(item_id="m0", text="candidate", metadata={}),),
        )
        for index in range(2)
    )
    method = _InvalidFirstMethod()

    with pytest.raises(ValueError, match="include every candidate"):
        run_retrieval(
            retrieval_method=method,
            requests=list(requests),
            top_k=1,
        )

    assert method.calls == 1
