from __future__ import annotations

from pathlib import Path
from typing import cast

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.evaluation.suites import session_recall_at
from graph_memory.evaluation.tables import metric_columns_for_rows
from graph_memory.registry.stage_configs import EvaluateIO, EvaluateStageConfig
from graph_memory.stages.evaluate import run_evaluate_stage


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


def test_longmemeval_metric_rows_select_longmemeval_table_columns() -> None:
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

    main_columns, path_columns, efficiency_columns, wide_columns = metric_columns_for_rows(result.metric_rows)

    assert "Turn Recall@5" in main_columns
    assert "Session Recall@10" in main_columns
    assert "Evidence F1@10" not in wide_columns
    assert path_columns == ["Method", "Path Recall@10", "Edge Recall@10"]
    assert "Memory Size" in efficiency_columns