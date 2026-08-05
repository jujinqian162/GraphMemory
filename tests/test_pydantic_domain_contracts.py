from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonNegativeInt,
    StrictBool,
)
from graph_memory.evaluation.requests import EvidenceLabel
from graph_memory.retrieval.contracts import RetrievalMethodResult
from graph_memory.retrieval.execution.service import run_retrieval
from graph_memory.retrieval.methods.ids import RetrievalMethodId
from graph_memory.retrieval.requests import (
    RankingMethodRequest,
    TextCandidate,
    TextRankingRequest,
)
from graph_memory.retrieval.results import RankedResult, RetrievedSubgraph
from graph_memory.stages.train_payloads import DenseFinetuneTrainPayload
from graph_memory.training_pairs.contracts import TrainPairRecord


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


def test_every_authoritative_retrieval_method_id_is_accepted_by_result_model() -> None:
    for method in RetrievalMethodId:
        result = RankedResult(
            task_id="task",
            method=method,
            ranked_nodes=(),
            retrieved_subgraph=RetrievedSubgraph(nodes=(), edges=()),
            latency_ms=0.0,
            input_tokens=0,
        )
        assert result.method is method


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


def test_training_payload_rejects_pair_drift_at_construction_boundary(
    tmp_path: Path,
) -> None:
    request = TextRankingRequest(
        task_id="task",
        query_text="query",
        candidates=(TextCandidate(item_id="m0", text="candidate", metadata={}),),
    )
    label = EvidenceLabel(
        task_id="task",
        gold_answer="answer",
        gold_evidence_item_ids=("m0",),
        gold_dependency_edges=(),
    )

    with pytest.raises(ValidationError, match="negative node is gold evidence"):
        DenseFinetuneTrainPayload(
            train_requests=(request,),
            train_labels=(label,),
            train_pairs=(
                TrainPairRecord(
                    task_id="task",
                    node_id="m0",
                    label=0,
                    sample_type="hard_dense",
                ),
            ),
            dev_requests=(request,),
            dev_labels=(label,),
            output_dir=tmp_path / "output",
            model_dir=tmp_path / "model",
        )
