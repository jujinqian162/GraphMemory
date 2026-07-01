from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.retrieval.requests import RealTimeRecencySpec
from graph_memory.datasets.longmemeval import (
    LongMemEvalLabelRecord,
    LongMemEvalRankingRecord,
    LongMemEvalToEvidenceEvaluationRequest,
    LongMemEvalToGraphBuildRequest,
    LongMemEvalToGraphRankingRequest,
    LongMemEvalToTemporalMemoryRankingRequest,
    LongMemEvalToTextRankingRequest,
)


def _ranking_record() -> LongMemEvalRankingRecord:
    return {
        "task_id": "longmem_q1",
        "question": "Where did I say I planned to meet Alex?",
        "question_datetime": "2024-01-10T12:00:00",
        "candidate_items": [
            {
                "item_id": "m0",
                "session_id": "s1",
                "session_order": 0,
                "turn_index": 0,
                "global_position": 0,
                "role": "user",
                "datetime": "2024-01-01T09:00:00",
                "text": "Let's meet Alex at the library tomorrow.",
            },
            {
                "item_id": "m1",
                "session_id": "s1",
                "session_order": 0,
                "turn_index": 1,
                "global_position": 1,
                "role": "assistant",
                "datetime": "2024-01-01T09:00:00",
                "text": "That sounds good.",
            },
        ],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "q1",
            "question_type": "single-session-user",
            "candidate_granularity": "turn",
        },
    }


def _label_record() -> LongMemEvalLabelRecord:
    return {
        "task_id": "longmem_q1",
        "gold_answer": "At the library.",
        "gold_support_item_ids": ["m0"],
        "gold_support_session_ids": ["s1"],
        "gold_dependency_edges": [],
        "metadata": {
            "dataset": "longmemeval_v1",
            "raw_id": "q1",
            "question_type": "single-session-user",
            "support_label_source": "has_answer",
        },
    }


def _graph() -> MemoryGraph:
    return {
        "task_id": "longmem_q1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Where?"},
            {"id": "m0", "node_type": "graph_item", "node_kind": "conversation_turn", "text": "Library."},
            {"id": "m1", "node_type": "graph_item", "node_kind": "conversation_turn", "text": "OK."},
        ],
        "edges": [],
    }


def test_longmemeval_text_projection_outputs_retriever_request_only() -> None:
    request = LongMemEvalToTextRankingRequest().project(_ranking_record())

    assert request.task_id == "longmem_q1"
    assert request.query_text == _ranking_record()["question"]
    assert request.candidates[0].item_id == "m0"
    assert request.candidates[0].text == "Let's meet Alex at the library tomorrow."
    assert request.candidates[0].metadata["session_id"] == "s1"
    assert request.candidates[0].metadata["session_order"] == 0
    assert request.candidates[0].metadata["sequence_index"] == 0
    assert request.candidates[0].metadata["position"] == 0


def test_longmemeval_temporal_projection_uses_real_time_recency_and_zero_importance() -> None:
    request = LongMemEvalToTemporalMemoryRankingRequest().project(_ranking_record())

    assert request.importance_by_item_id == {"m0": 0.0, "m1": 0.0}
    assert request.metadata["recency_mode"] == "real_time"
    assert request.metadata["question_datetime"] == "2024-01-10T12:00:00"
    assert request.metadata["position_by_item_id"] == {"m0": 0, "m1": 1}
    assert request.metadata["session_order_by_item_id"] == {"m0": 0, "m1": 0}
    assert request.metadata["turn_index_by_item_id"] == {"m0": 0, "m1": 1}
    assert request.metadata["datetime_by_item_id"] == {
        "m0": "2024-01-01T09:00:00",
        "m1": "2024-01-01T09:00:00",
    }
    assert isinstance(request.recency, RealTimeRecencySpec)
    assert request.recency.question_datetime == "2024-01-10T12:00:00"
    assert request.recency.datetime_by_item_id == {
        "m0": "2024-01-01T09:00:00",
        "m1": "2024-01-01T09:00:00",
    }


def test_longmemeval_graph_projection_uses_session_local_sequence_index() -> None:
    request = LongMemEvalToGraphBuildRequest().project(_ranking_record())

    assert request.task_id == "longmem_q1"
    assert request.nodes[0].node_id == "m0"
    assert request.nodes[0].node_kind == "conversation_turn"
    assert request.nodes[0].source_ref == "s1"
    assert request.nodes[0].group_key == "session:s1"
    assert request.nodes[0].sequence_index == 0
    assert request.nodes[0].metadata["global_position"] == 0
    assert request.input_visible_edges == ()


def test_longmemeval_graph_ranking_projection_reuses_text_candidates() -> None:
    request = LongMemEvalToGraphRankingRequest().project(_ranking_record(), _graph(), {"m0": 0.7})

    assert request.task_id == "longmem_q1"
    assert request.graph["task_id"] == "longmem_q1"
    assert request.initial_scores == {"m0": 0.7}
    assert [candidate.item_id for candidate in request.candidates] == ["m0", "m1"]


def test_longmemeval_evaluation_projection_outputs_turn_support_labels() -> None:
    ranked_result: RankedResult = {
        "task_id": "longmem_q1",
        "method": "bm25",
        "ranked_nodes": [{"node_id": "m0", "score": 2.0}, {"node_id": "m1", "score": 1.0}],
        "retrieved_subgraph": {"nodes": [], "edges": []},
        "latency_ms": 1.0,
        "input_tokens": 8,
    }
    request = LongMemEvalToEvidenceEvaluationRequest().project(
        predictions=[ranked_result],
        labels=[_label_record()],
        graphs=[_graph()],
    )

    assert request.labels[0].task_id == "longmem_q1"
    assert request.labels[0].gold_evidence_item_ids == ("m0",)
    assert request.labels[0].gold_dependency_edges == ()
