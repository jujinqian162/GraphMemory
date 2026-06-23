from __future__ import annotations

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig
from graph_memory.retrieval.methods.fast_graphrag.index import (
    build_fast_graphrag_knowledge_graph,
    official_noun_graph_snapshot,
)
from graph_memory.retrieval.requests import TextCandidate, TextRankingRequest


def test_official_noun_graph_snapshot_exposes_entities_and_relationships() -> None:
    request = TextRankingRequest(
        task_id="task-1",
        query_text="Richmond census",
        candidates=(
            TextCandidate("m0", "Richmond census-designated place in Maine.", {"position": 0}),
            TextCandidate("m1", "Richmond population census result.", {"position": 1}),
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
        "edges": [],
    }
    kg = build_fast_graphrag_knowledge_graph(request, graph, config=FastGraphRAGConfig())

    snapshot = official_noun_graph_snapshot(kg)

    assert set(snapshot) == {"entities", "relationships"}
    assert snapshot["entities"]
    assert all(
        set(row) == {"title", "frequency", "text_unit_ids", "type", "description"}
        for row in snapshot["entities"]
    )
    assert all(
        set(row) == {"source", "target", "weight", "text_unit_ids", "description"}
        for row in snapshot["relationships"]
    )
