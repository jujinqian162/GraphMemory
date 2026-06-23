from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.config import FastGraphRAGScoringConfig
from graph_memory.retrieval.methods.fast_graphrag.scoring import score_candidates
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


def test_candidate_scores_do_not_reward_repeated_mentions_without_relation_support() -> None:
    candidates = (
        TextCandidate(item_id="m0", text="A mentions B once.", metadata={"position": 0}),
        TextCandidate(item_id="m1", text="A repeats A A A.", metadata={"position": 1}),
    )
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:a", "A", "a", "noun_phrase", "A", ("m0", "m1")),
            FastGraphRAGEntity("e:b", "B", "b", "noun_phrase", "B", ("m0",)),
        ),
        relations=(
            FastGraphRAGRelation("r:a:b", "e:a", "e:b", "co-occurs", ("m0",), 1.0),
        ),
    )

    scores = score_candidates(
        candidates,
        kg,
        entity_scores={"e:a": 0.5, "e:b": 0.5},
        dense_fallback_scores={"m0": 0.0, "m1": 0.0},
        config=FastGraphRAGScoringConfig(lambda_entity=1.0, lambda_relation=1.0, lambda_dense_fallback=0.0),
    )

    assert scores["m0"] > scores["m1"]


def test_candidate_scores_normalize_many_weak_relations_per_candidate() -> None:
    candidates = (
        TextCandidate(item_id="m0", text="Donna Tartt was born in 1963.", metadata={"position": 0}),
        TextCandidate(item_id="m1", text="Donna Tartt appears with many unrelated phrases.", metadata={"position": 1}),
    )
    noisy_entities = tuple(
        FastGraphRAGEntity(f"e:noise-{index}", f"Noise {index}", f"noise {index}", "noun_phrase", "", ("m1",))
        for index in range(8)
    )
    noisy_relations = tuple(
        FastGraphRAGRelation(f"r:donna:noise-{index}", "e:donna-tartt", f"e:noise-{index}", "co-occurs", ("m1",), 1.0)
        for index in range(8)
    )
    kg = FastGraphRAGKnowledgeGraph(
        entities=(
            FastGraphRAGEntity("e:donna-tartt", "Donna Tartt", "donna tartt", "noun_phrase", "", ("m0", "m1")),
            FastGraphRAGEntity("e:born", "born", "born", "noun_phrase", "", ("m0",)),
            *noisy_entities,
        ),
        relations=(
            FastGraphRAGRelation("r:donna:born", "e:donna-tartt", "e:born", "co-occurs", ("m0",), 1.0),
            *noisy_relations,
        ),
    )

    scores = score_candidates(
        candidates,
        kg,
        entity_scores={"e:donna-tartt": 1.0, "e:born": 0.8},
        dense_fallback_scores={"m0": 0.0, "m1": 0.0},
        config=FastGraphRAGScoringConfig(lambda_entity=1.0, lambda_relation=1.0, lambda_dense_fallback=0.0),
    )

    assert scores["m0"] > scores["m1"]
