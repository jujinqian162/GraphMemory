from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from graph_memory.contracts.graphs import MemoryGraph
from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGConfig
from graph_memory.retrieval.methods.fast_graphrag.method import FastGraphRAGMethod
from graph_memory.retrieval.requests import (
    FastGraphRAGEntity,
    FastGraphRAGKnowledgeGraph,
    FastGraphRAGRelation,
    FastGraphRAGRequest,
    TextCandidate,
    TextRankingRequest,
)


class FakeDenseSeedRanker:
    def __init__(
        self,
        *,
        entity_scores: Mapping[str, float] | None = None,
        candidate_scores: Mapping[str, float] | None = None,
    ) -> None:
        self.entity_scores = dict(entity_scores or {})
        self.candidate_scores = dict(candidate_scores or {})

    def score_entities(self, query_text: str, entities: Sequence[FastGraphRAGEntity]) -> Mapping[str, float]:
        _ = query_text
        return {entity.entity_id: self.entity_scores.get(entity.entity_id, 0.0) for entity in entities}

    def score_candidates(self, query_text: str, candidates: Sequence[TextCandidate]) -> Mapping[str, float]:
        _ = query_text
        return {candidate.item_id: self.candidate_scores.get(candidate.item_id, 0.0) for candidate in candidates}


def test_fast_graphrag_method_returns_full_ranking_and_topk_candidate_edges() -> None:
    request = fast_graphrag_fixture_request()
    method = FastGraphRAGMethod(
        name="fast_graphrag",
        config=FastGraphRAGConfig(),
        dense_ranker=FakeDenseSeedRanker(
            entity_scores={"e:changed-it": 0.9, "e:nicki-minaj": 0.7},
            candidate_scores={"m0": 0.8, "m1": 0.6, "m2": 0.1},
        ),
    )

    result = method.rank_task(request, top_k=2)

    assert [node.node_id for node in result.ranked_nodes] == ["m0", "m1", "m2"]
    assert result.trace.retrieved_edges == [
        {"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 1.0, "directed": False}
    ]


def test_fast_graphrag_method_rejects_non_fast_graphrag_request() -> None:
    method = FastGraphRAGMethod(name="fast_graphrag", config=FastGraphRAGConfig(), dense_ranker=FakeDenseSeedRanker())

    with pytest.raises(TypeError, match="FastGraphRAGRequest"):
        method.rank_task(TextRankingRequest(task_id="x", query_text="q", candidates=()), top_k=1)


def test_fast_graphrag_query_linked_entities_override_dense_noise() -> None:
    request = fast_graphrag_fixture_request()
    method = FastGraphRAGMethod(
        name="fast_graphrag",
        config=FastGraphRAGConfig(query_link_seed_score=1.0, dense_entity_seed_weight=0.1),
        dense_ranker=FakeDenseSeedRanker(
            entity_scores={"e:changed-it": 0.0, "e:nicki-minaj": 0.1},
            candidate_scores={"m0": 0.0, "m1": 0.0, "m2": 0.9},
        ),
    )

    result = method.rank_task(request, top_k=2)

    assert result.ranked_nodes[0].node_id in {"m0", "m1"}


def fast_graphrag_fixture_request() -> FastGraphRAGRequest:
    candidates = (
        TextCandidate(item_id="m0", text="Changed It is a song by Nicki Minaj.", metadata={"position": 0}),
        TextCandidate(item_id="m1", text="Nicki Minaj performed Changed It.", metadata={"position": 1}),
        TextCandidate(item_id="m2", text="Unrelated.", metadata={"position": 2}),
    )
    candidate_graph: MemoryGraph = {
        "task_id": "task-1",
        "nodes": [
            {"id": "q", "node_type": "question", "text": "Who performed Changed It?"},
            {"id": "m0", "node_type": "graph_item", "node_kind": "document_sentence", "text": candidates[0].text},
            {"id": "m1", "node_type": "graph_item", "node_kind": "document_sentence", "text": candidates[1].text},
            {"id": "m2", "node_type": "graph_item", "node_kind": "document_sentence", "text": candidates[2].text},
        ],
        "edges": [{"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 1.0, "directed": False}],
    }
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:changed-it", "Changed It", "changed it", "document_title", "Changed It", ("m0",)),
            FastGraphRAGEntity("e:nicki-minaj", "Nicki Minaj", "nicki minaj", "mention", "Nicki Minaj", ("m0", "m1")),
        ),
        relations=(
            FastGraphRAGRelation("r:changed:nicki:m0", "e:changed-it", "e:nicki-minaj", "co-occurs", ("m0",), 1.0),
        ),
    )
    return FastGraphRAGRequest(
        task_id="task-1",
        query_text="Who performed Changed It?",
        candidates=candidates,
        candidate_graph=candidate_graph,
        knowledge_graph=kg,
    )
