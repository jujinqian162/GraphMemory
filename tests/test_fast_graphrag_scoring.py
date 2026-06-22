from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.scoring import (
    FastGraphRAGScoringConfig,
    score_candidates,
)
from graph_memory.retrieval.requests import (
    FastGraphRAGEntity,
    FastGraphRAGKnowledgeGraph,
    FastGraphRAGRelation,
    TextCandidate,
)


def test_candidate_scores_aggregate_entity_and_relation_support() -> None:
    candidates = (
        TextCandidate(item_id="m0", text="Changed It mentions Nicki Minaj.", metadata={"position": 0}),
        TextCandidate(item_id="m1", text="Unrelated.", metadata={"position": 1}),
    )
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:changed-it", "Changed It", "changed it", "document_title", "Changed It", ("m0",)),
            FastGraphRAGEntity("e:nicki-minaj", "Nicki Minaj", "nicki minaj", "mention", "Nicki Minaj", ("m0",)),
        ),
        relations=(
            FastGraphRAGRelation("r:changed:nicki:m0", "e:changed-it", "e:nicki-minaj", "co-occurs", ("m0",), 1.0),
        ),
    )

    scores = score_candidates(
        candidates,
        kg,
        entity_scores={"e:changed-it": 0.6, "e:nicki-minaj": 0.4},
        dense_fallback_scores={"m0": 0.2, "m1": 0.1},
        config=FastGraphRAGScoringConfig(lambda_entity=1.0, lambda_relation=1.0, lambda_dense_fallback=0.05),
    )

    assert scores["m0"] > scores["m1"]
