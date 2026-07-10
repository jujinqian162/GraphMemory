from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.contracts.ranking import RankedResult
from graph_memory.datasets.musique import (
    MuSiQueLabelRecord,
    MuSiQueRankingRecord,
    MuSiQueToEvidenceEvaluationRequest,
    MuSiQueToGraphBuildRequest,
    MuSiQueToGraphRankingRequest,
    MuSiQueToTemporalMemoryRankingRequest,
    MuSiQueToTextRankingRequest,
    convert_musique_example,
    parse_musique_example,
)


def _raw_example() -> dict[str, object]:
    return {
        "id": "2hop__1_2",
        "question": "Where was the director of Film A born?",
        "answer": "London",
        "answer_aliases": ["London, England"],
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "Film A", "paragraph_text": "Film A was directed by Ada.", "is_supporting": True},
            {
                "idx": 1,
                "title": "Ada Lovelace",
                "paragraph_text": "Ada Lovelace was born in London.",
                "is_supporting": True,
            },
            {"idx": 2, "title": "Distractor", "paragraph_text": "This paragraph is not useful.", "is_supporting": False},
        ],
        "question_decomposition": [
            {"id": "1", "question": "Who directed Film A?", "answer": "Ada Lovelace", "paragraph_support_idx": 0},
            {"id": "2", "question": "Where was #1 born?", "answer": "London", "paragraph_support_idx": 1},
        ],
    }


def _ranking_record() -> MuSiQueRankingRecord:
    return convert_musique_example(parse_musique_example(_raw_example())).ranking_record


def _label_record() -> MuSiQueLabelRecord:
    return convert_musique_example(parse_musique_example(_raw_example())).label_record


def _graph() -> MemoryGraph:
    return {
        "task_id": "musique_2hop__1_2",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Where?"},
            {"id": "p0", "node_type": "graph_item", "node_kind": "document_paragraph", "text": "Film A."},
            {"id": "p1", "node_type": "graph_item", "node_kind": "document_paragraph", "text": "Ada."},
        ],
        "edges": [{"source": "p0", "target": "p1", "edge_type": "bridge", "weight": 1.0, "directed": False}],
    }


def test_musique_conversion_outputs_paragraph_labels_and_dependency_edges() -> None:
    converted = convert_musique_example(parse_musique_example(_raw_example()))

    assert converted.ranking_record["task_id"] == "musique_2hop__1_2"
    assert [paragraph["paragraph_id"] for paragraph in converted.ranking_record["candidate_paragraphs"]] == ["p0", "p1", "p2"]
    assert converted.label_record["gold_evidence_paragraph_ids"] == ["p0", "p1"]
    assert converted.label_record["gold_dependency_edges"] == [["p0", "p1"]]
    assert converted.label_record["metadata"]["path_supported"] is True
    assert "answer" not in converted.ranking_record["candidate_paragraphs"][0]
    assert "is_supporting" not in converted.ranking_record["candidate_paragraphs"][0]


def test_musique_text_projection_outputs_retriever_request_only() -> None:
    request = MuSiQueToTextRankingRequest().project(_ranking_record())

    assert request.task_id == "musique_2hop__1_2"
    assert request.query_text == "Where was the director of Film A born?"
    assert request.candidates[0].item_id == "p0"
    assert request.candidates[0].text == "Film A. Film A was directed by Ada."
    assert request.candidates[0].metadata == {
        "title": "Film A",
        "source_ref": "Film A",
        "sequence_index": 0,
        "position": 0,
    }


def test_musique_graph_projection_uses_only_input_visible_fields() -> None:
    request = MuSiQueToGraphBuildRequest().project(_ranking_record())

    assert request.task_id == "musique_2hop__1_2"
    assert request.nodes[0].node_id == "p0"
    assert request.nodes[0].node_kind == "document_paragraph"
    assert request.nodes[0].source_ref == "Film A"
    assert request.nodes[0].group_key == "document:Film A"
    assert request.nodes[0].metadata == {"title": "Film A", "position": 0}
    assert request.input_visible_edges == ()


def test_musique_graph_ranking_projection_reuses_text_candidates() -> None:
    request = MuSiQueToGraphRankingRequest().project(_ranking_record(), _graph(), {"p0": 0.7})

    assert request.task_id == "musique_2hop__1_2"
    assert request.graph["task_id"] == "musique_2hop__1_2"
    assert request.initial_scores == {"p0": 0.7}
    assert [candidate.item_id for candidate in request.candidates] == ["p0", "p1", "p2"]


def test_musique_temporal_projection_uses_paragraph_positions() -> None:
    request = MuSiQueToTemporalMemoryRankingRequest().project(_ranking_record(), {"p0": 0.5})

    assert request.importance_by_item_id == {"p0": 0.5}
    assert request.metadata == {"position_by_item_id": {"p0": 0, "p1": 1, "p2": 2}}


def test_musique_evaluation_projection_outputs_dependency_edges() -> None:
    ranked_result: RankedResult = {
        "task_id": "musique_2hop__1_2",
        "method": "dense_graph_rerank",
        "ranked_nodes": [{"node_id": "p0", "score": 2.0}, {"node_id": "p1", "score": 1.0}],
        "retrieved_subgraph": {
            "nodes": ["p0", "p1"],
            "edges": [{"source": "p0", "target": "p1", "edge_type": "bridge", "weight": 1.0, "directed": False}],
        },
        "latency_ms": 1.0,
        "input_tokens": 8,
    }
    request = MuSiQueToEvidenceEvaluationRequest().project(
        predictions=[ranked_result],
        labels=[_label_record()],
        graphs=[_graph()],
    )

    assert request.labels[0].task_id == "musique_2hop__1_2"
    assert request.labels[0].gold_evidence_item_ids == ("p0", "p1")
    assert request.labels[0].gold_dependency_edges == (("p0", "p1"),)
