from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_type_hints

import pytest

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.metrics import EvidenceMetricRow, LongMemEvalMetricRow
from graph_memory.contracts.ranking import RankedResult
from graph_memory.evaluation.requests import EvidenceEvaluationRequest, EvidenceLabel
from graph_memory.evaluation.suites import EvidenceMetricSuite, LongMemEvalMetricSuite, session_recall_at
from graph_memory.registry.stage_configs import EvaluateIO, EvaluateStageConfig
from graph_memory.retrieval.methods.graph_rerank.config import GraphRerankConfig
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest
from graph_memory.retrieval.tuning import graph_rerank as graph_rerank_tuning
from graph_memory.retrieval.tuning.selection import longmemeval_retrieval_candidate_key
from graph_memory.stages.evaluate import run_evaluate_stage
from scripts import tune_graph_rerank as tune_graph_rerank_script


def _prediction() -> RankedResult:
    return cast(
        RankedResult,
        cast(
            object,
            {
                "task_id": "longmem_q1",
                "method": "memory_stream",
                "ranked_nodes": [
                    {"node_id": "m2", "score": 2.0},
                    {"node_id": "m0", "score": 1.0},
                ],
                "retrieved_subgraph": {"nodes": [], "edges": []},
                "latency_ms": 12.0,
            },
        ),
    )

def _graph() -> MemoryGraph:
    return cast(
        MemoryGraph,
        cast(
            object,
            {
                "task_id": "longmem_q1",
                "nodes": [
                    {"id": "q", "node_type": "question", "text": "Where did I plan to meet Alex?"},
                    {
                        "id": "m0",
                        "node_type": "graph_item",
                        "node_kind": "conversation_turn",
                        "text": "Meet Alex at the library.",
                        "metadata": {"session_id": "s1"},
                    },
                    {
                        "id": "m1",
                        "node_type": "graph_item",
                        "node_kind": "conversation_turn",
                        "text": "Unrelated memory.",
                        "metadata": {"session_id": "s2"},
                    },
                    {
                        "id": "m2",
                        "node_type": "graph_item",
                        "node_kind": "conversation_turn",
                        "text": "Another unrelated memory.",
                        "metadata": {"session_id": "s3"},
                    },
                ],
                "edges": [],
            },
        ),
    )

def _label() -> dict[str, object]:
    return {
        "task_id": "longmem_q1",
        "gold_answer": "At the library.",
        "gold_support_item_ids": ["m0"],
        "gold_support_session_ids": ["s1"],
        "gold_dependency_edges": [],
        "metadata": {"dataset": "longmemeval_v1"},
    }


def test_session_recall_at_counts_unique_retrieved_sessions() -> None:
    ranked_node_ids = ["m2", "m0"]
    item_to_session = {"m0": "s1", "m1": "s2", "m2": "s3"}

    assert session_recall_at(ranked_node_ids, item_to_session, {"s1"}, 2) == 1.0


def test_metric_suites_expose_distinct_row_contracts() -> None:
    assert get_type_hints(EvidenceMetricSuite.evaluate)["return"] == list[EvidenceMetricRow]
    assert get_type_hints(LongMemEvalMetricSuite.evaluate)["return"] == list[LongMemEvalMetricRow]


def test_graph_rerank_tuning_targets_use_longmemeval_suite() -> None:
    metric_suite, selection_key = tune_graph_rerank_script._graph_rerank_tuning_targets("longmemeval")

    assert metric_suite.name == "longmemeval"
    assert selection_key is longmemeval_retrieval_candidate_key


class _FakeLongMemEvalMetricSuite:
    name = "longmemeval"

    def evaluate(self, request: EvidenceEvaluationRequest) -> Sequence[LongMemEvalMetricRow]:
        score = float(request.predictions[0]["latency_ms"])
        return [
            {
                "Method": str(request.predictions[0]["method"]),
                "Turn Recall@5": 0.0,
                "Turn Recall@10": score,
                "Full Turn Support@10": score,
                "Session Recall@5": 0.0,
                "Session Recall@10": score,
                "Full Session Support@10": 0.0,
                "MRR": score,
                "Path Recall@10": "N/A",
                "Edge Recall@10": "N/A",
                "Retrieval Latency / Query": 1.0,
                "Memory Size": 1.0,
                "Avg Retrieved Nodes": 0.0,
                "Avg Retrieved Edges": 0.0,
            }
        ]


def test_graph_rerank_tuning_service_uses_injected_metric_suite_and_selection_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_graph_rerank_from_seed_score_cache(
        *,
        method: str,
        graph_config: GraphRerankConfig,
        **_: object,
    ) -> list[RankedResult]:
        return [
            cast(
                RankedResult,
                cast(
                    object,
                    {
                        "task_id": "longmem_q1",
                        "method": method,
                        "ranked_nodes": [{"node_id": "m0", "score": graph_config.lambda_init}],
                        "retrieved_subgraph": {"nodes": [], "edges": []},
                        "latency_ms": graph_config.lambda_init,
                    },
                ),
            )
        ]

    monkeypatch.setattr(graph_rerank_tuning, "precompute_seed_score_cache", lambda **_: object())
    monkeypatch.setattr(
        graph_rerank_tuning,
        "run_graph_rerank_from_seed_score_cache",
        fake_run_graph_rerank_from_seed_score_cache,
    )

    selected_config, candidate_rows = graph_rerank_tuning.tune_graph_rerank(
        method="dense_graph_rerank",
        ranking_requests=[
            TextRankingRequest(
                task_id="longmem_q1",
                query_text="Where did I plan to meet Alex?",
                candidates=(TextCandidate(item_id="m0", text="Meet Alex at the library.", metadata={}),),
            )
        ],
        labels=[
            EvidenceLabel(
                task_id="longmem_q1",
                gold_answer="At the library.",
                gold_evidence_item_ids=("m0",),
                gold_dependency_edges=(),
                gold_session_ids=("s1",),
            )
        ],
        graphs=[_graph()],
        grid=[GraphRerankConfig(lambda_init=0.0), GraphRerankConfig(lambda_init=1.0)],
        metric_suite=_FakeLongMemEvalMetricSuite(),
        selection_key=longmemeval_retrieval_candidate_key,
    )

    assert selected_config["lambda_init"] == 1.0
    assert cast(LongMemEvalMetricRow, candidate_rows[1])["Full Turn Support@10"] == 1.0


def test_longmemeval_evaluate_stage_outputs_turn_and_session_metrics() -> None:
    result = run_evaluate_stage(
        EvaluateStageConfig(
            io=EvaluateIO(
                predictions=Path("predictions.json"),
                labels=Path("labels.json"),
                graphs=Path("graphs.json"),
                output=Path("metrics.csv"),
                failure_cases_output=None,
            ),
            dataset="longmemeval",
        ),
        predictions=[_prediction()],
        labels=[_label()],
        graphs=[_graph()],
    )

    row = cast(dict[str, object], cast(object, result.metric_rows[0]))
    assert row["Method"] == "memory_stream"
    assert row["Turn Recall@5"] == 1.0
    assert row["Turn Recall@10"] == 1.0
    assert row["Full Turn Support@10"] == 1.0
    assert row["Session Recall@5"] == 1.0
    assert row["Session Recall@10"] == 1.0
    assert row["Full Session Support@10"] == 1.0
    assert row["MRR"] == 0.5
    assert row["Retrieval Latency / Query"] == 12.0
    assert row["Memory Size"] == 3.0
    assert "Evidence F1@10" not in row


def test_longmemeval_metric_rows_use_suite_owned_table_schema() -> None:
    result = run_evaluate_stage(
        EvaluateStageConfig(
            io=EvaluateIO(
                predictions=Path("predictions.json"),
                labels=Path("labels.json"),
                graphs=Path("graphs.json"),
                output=Path("metrics.csv"),
                failure_cases_output=None,
            ),
            dataset="longmemeval",
        ),
        predictions=[_prediction()],
        labels=[_label()],
        graphs=[_graph()],
    )

    schema = result.metric_table_schema

    assert result.metric_suite_name == "longmemeval"
    assert "Turn Recall@5" in schema.main_columns
    assert "Session Recall@10" in schema.main_columns
    assert "Evidence F1@10" not in schema.wide_columns
    assert list(schema.path_columns) == ["Method", "Path Recall@10", "Edge Recall@10"]
    assert "Memory Size" in schema.efficiency_columns
