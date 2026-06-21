from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.twowiki import (
    TwoWikiLabelRecord,
    TwoWikiRankingRecord,
    TwoWikiToEvidenceEvaluationRequest,
    TwoWikiToGraphBuildRequest,
    TwoWikiToGraphRankingRequest,
    TwoWikiToTemporalMemoryRankingRequest,
    TwoWikiToTextRankingRequest,
)


def _ranking_record() -> TwoWikiRankingRecord:
    return {
        "task_id": "2wiki_abc123",
        "question": "Who is Ada's mother?",
        "question_type": "compositional",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "title": "Film A",
                "sentence_index": 0,
                "position": 0,
                "text": "Film A was directed by Ada.",
            },
            {
                "sentence_id": "m1",
                "title": "Ada Lovelace",
                "sentence_index": 0,
                "position": 1,
                "text": "Ada was the daughter of Beth.",
            },
        ],
        "metadata": {"dataset": "2wiki", "raw_id": "abc123"},
    }


def _label_record() -> TwoWikiLabelRecord:
    return {
        "task_id": "2wiki_abc123",
        "gold_answer": "Beth",
        "gold_evidence_sentence_ids": ["m0", "m1"],
        "gold_dependency_edges": [["m0", "m1"]],
        "metadata": {
            "question_type": "compositional",
            "path_label_source": "evidences",
            "path_supported": True,
            "mapping_ambiguity_count": 0,
        },
    }


def _graph() -> MemoryGraph:
    return {
        "task_id": "2wiki_abc123",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Who?"},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": "A"},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": "B"},
        ],
        "edges": [{"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False}],
    }


def test_twowiki_text_projection_outputs_retriever_request_only() -> None:
    request = TwoWikiToTextRankingRequest().project(_ranking_record())

    assert request.task_id == "2wiki_abc123"
    assert request.query_text == "Who is Ada's mother?"
    assert request.candidates[0].item_id == "m0"
    assert request.candidates[0].text == "Film A. Film A was directed by Ada."
    assert request.candidates[0].metadata == {
        "title": "Film A",
        "source_ref": "Film A",
        "sequence_index": 0,
        "position": 0,
        "question_type": "compositional",
    }


def test_twowiki_graph_projection_uses_only_input_visible_fields() -> None:
    request = TwoWikiToGraphBuildRequest().project(_ranking_record())

    assert request.task_id == "2wiki_abc123"
    assert request.nodes[0].node_id == "m0"
    assert request.nodes[0].source_ref == "Film A"
    assert request.nodes[0].group_key == "document:Film A"
    assert request.nodes[0].metadata == {
        "title": "Film A",
        "position": 0,
        "question_type": "compositional",
    }
    assert request.input_visible_edges == ()


def test_twowiki_graph_ranking_projection_reuses_text_candidates() -> None:
    request = TwoWikiToGraphRankingRequest().project(_ranking_record(), _graph(), {"m0": 0.7})

    assert request.task_id == "2wiki_abc123"
    assert request.graph["task_id"] == "2wiki_abc123"
    assert request.initial_scores == {"m0": 0.7}
    assert [candidate.item_id for candidate in request.candidates] == ["m0", "m1"]


def test_twowiki_temporal_projection_uses_synthetic_positions_only() -> None:
    request = TwoWikiToTemporalMemoryRankingRequest().project(_ranking_record(), {"m0": 0.5})

    assert request.importance_by_item_id == {"m0": 0.5}
    assert request.metadata == {"position_by_item_id": {"m0": 0, "m1": 1}}


def test_twowiki_evaluation_projection_outputs_dependency_edges() -> None:
    ranked_result: RankedResult = {
        "task_id": "2wiki_abc123",
        "method": "dense_graph_rerank",
        "ranked_nodes": [{"node_id": "m0", "score": 2.0}, {"node_id": "m1", "score": 1.0}],
        "retrieved_subgraph": {
            "nodes": ["m0", "m1"],
            "edges": [{"source": "m0", "target": "m1", "edge_type": "bridge", "weight": 1.0, "directed": False}],
        },
        "latency_ms": 1.0,
        "input_tokens": 8,
    }
    request = TwoWikiToEvidenceEvaluationRequest().project(
        predictions=[ranked_result],
        labels=[_label_record()],
        graphs=[_graph()],
    )

    assert request.labels[0].task_id == "2wiki_abc123"
    assert request.labels[0].gold_evidence_item_ids == ("m0", "m1")
    assert request.labels[0].gold_dependency_edges == (("m0", "m1"),)
