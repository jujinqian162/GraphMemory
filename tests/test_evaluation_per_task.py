from __future__ import annotations

import statistics

from graph_memory.retrieval.results import RankedResult
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.suites import evidence_metric_suite


def _prediction(task_id: str, ranked: list[str]) -> RankedResult:
    return RankedResult.model_validate({
        "task_id": task_id,
        "method": "bm25",
        "ranked_nodes": [
            {"node_id": node_id, "score": float(len(ranked) - index)}
            for index, node_id in enumerate(ranked)
        ],
        "retrieved_subgraph": {"nodes": list(ranked), "edges": []},
        "latency_ms": 2.0,
        "input_tokens": 10,
    })


def _label(task_id: str, gold: list[str]) -> EvidenceLabel:
    return EvidenceLabel(
        task_id=task_id,
        gold_answer="a",
        gold_evidence_item_ids=tuple(gold),
        gold_dependency_edges=(),
    )


def _request() -> EvidenceEvaluationRequest:
    predictions = [
        _prediction("t1", ["m0", "m1", "m2"]),
        _prediction("t2", ["m9", "m0", "m1"]),
    ]
    labels = [
        _label("t1", ["m0", "m1"]),
        _label("t2", ["m0", "m1"]),
    ]
    return EvidenceEvaluationRequest(
        predictions=tuple(predictions), labels=tuple(labels), graphs=()
    )


def test_per_task_rows_are_keyed_by_task_id_matching_predictions() -> None:
    suite = evidence_metric_suite()
    _, per_task = suite.evaluate_with_per_task(_request())

    task_ids = [row.task_id for row in per_task]
    assert task_ids == ["t1", "t2"]
    assert len(set(task_ids)) == len(task_ids)


def test_per_task_metrics_average_to_aggregate() -> None:
    suite = evidence_metric_suite()
    aggregate, per_task = suite.evaluate_with_per_task(_request())

    for metric in ("Recall@2", "Recall@5", "Recall@10", "MRR", "Full Support@5"):
        averaged = statistics.fmean(
            float(row.model_dump(mode="json", by_alias=True)[metric])
            for row in per_task
        )
        assert averaged == aggregate[0].model_dump(mode="json", by_alias=True)[metric]


def test_plain_evaluate_still_returns_only_aggregate() -> None:
    suite = evidence_metric_suite()
    rows = suite.evaluate(_request())
    assert len(rows) == 1
    assert "task_id" not in type(rows[0]).model_fields
