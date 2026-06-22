from __future__ import annotations

import pytest

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.retrieval.methods.fast_graphrag.index import build_fast_graphrag_knowledge_graph
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest
from graph_memory.validation import ContractValidationError


def test_build_fast_graphrag_kg_uses_candidate_titles_mentions_and_visible_relations() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Who performed Changed It?",
        candidates=(
            TextCandidate(
                item_id="m0",
                text="Changed It is a song by Nicki Minaj.",
                metadata={"title": "Changed It", "position": 0},
            ),
            TextCandidate(
                item_id="m1",
                text="Nicki Minaj was born in Trinidad and Tobago.",
                metadata={"title": "Nicki Minaj", "position": 1},
            ),
        ),
    )
    graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": request.candidates[0].text,
            },
            {
                "id": "m1",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": request.candidates[1].text,
            },
        ],
        "edges": [{"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 1.0, "directed": False}],
    }

    kg = build_fast_graphrag_knowledge_graph(request, graph)

    names = {entity.name for entity in kg.entities}
    assert "Changed It" in names
    assert "Nicki Minaj" in names
    relation = next(relation for relation in kg.relations if relation.candidate_ids == ("m0",))
    assert "Changed It is a song by Nicki Minaj." in relation.description


def test_build_fast_graphrag_kg_rejects_stale_graph_text() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Who performed Changed It?",
        candidates=(
            TextCandidate(
                item_id="m0",
                text="Changed It is a song by Nicki Minaj.",
                metadata={"title": "Changed It", "position": 0},
            ),
        ),
    )
    graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "Stale graph text for the same candidate id.",
            },
        ],
        "edges": [],
    }

    with pytest.raises(ContractValidationError, match="graph node text mismatch"):
        build_fast_graphrag_knowledge_graph(request, graph)


def test_build_fast_graphrag_kg_accepts_title_prefixed_candidate_text() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Who performed Changed It?",
        candidates=(
            TextCandidate(
                item_id="m0",
                text="Changed It. Changed It is a song by Nicki Minaj.",
                metadata={"title": "Changed It", "position": 0},
            ),
        ),
    )
    graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": request.query_text},
            {
                "id": "m0",
                "node_type": "graph_item",
                "node_kind": "document_sentence",
                "text": "Changed It is a song by Nicki Minaj.",
                "source_ref": "Changed It",
            },
        ],
        "edges": [],
    }

    kg = build_fast_graphrag_knowledge_graph(request, graph)

    assert [entity.name for entity in kg.entities] == ["Changed It", "Nicki Minaj"]
