from __future__ import annotations

from graph_memory.retrieval.requests import (
    FastGraphRAGEntity,
    FastGraphRAGKnowledgeGraph,
    FastGraphRAGRelation,
    FastGraphRAGRequest,
    RankingMethodRequest,
    TextCandidate,
)


def test_fast_graphrag_request_carries_visible_candidate_graph_and_kg() -> None:
    candidate = TextCandidate(item_id="m0", text="Paris is in France.", metadata={"title": "Paris"})
    entity = FastGraphRAGEntity(
        entity_id="e:paris",
        name="Paris",
        normalized_name="paris",
        entity_type="document_title",
        description="Paris",
        candidate_ids=("m0",),
    )
    relation = FastGraphRAGRelation(
        relation_id="r:e:paris:e:france:m0",
        source_entity_id="e:paris",
        target_entity_id="e:france",
        description="Paris co-occurs with France in m0.",
        candidate_ids=("m0",),
        weight=1.0,
    )
    request = FastGraphRAGRequest(
        task_id="task-1",
        query_text="Where is Paris?",
        candidates=(candidate,),
        candidate_graph={
            "task_id": "task-1",
            "nodes": [{"id": "q", "node_type": "question", "text": "Where is Paris?"}],
            "edges": [],
        },
        knowledge_graph=FastGraphRAGKnowledgeGraph(entities=(entity,), relations=(relation,)),
    )

    typed_request: RankingMethodRequest = request
    assert typed_request.task_id == "task-1"
    assert request.knowledge_graph.entities[0].candidate_ids == ("m0",)
