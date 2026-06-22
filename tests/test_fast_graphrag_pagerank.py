from __future__ import annotations

from graph_memory.retrieval.methods.fast_graphrag.pagerank import personalized_pagerank


def test_personalized_pagerank_prefers_seed_connected_nodes() -> None:
    adjacency = {
        "a": {"b": 1.0},
        "b": {"a": 1.0, "c": 1.0},
        "c": {"b": 1.0},
    }

    scores = personalized_pagerank(
        adjacency,
        {"a": 1.0},
        damping=0.85,
        max_iterations=100,
        tolerance=1e-8,
    )

    assert scores["a"] > scores["c"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_personalized_pagerank_handles_dangling_nodes() -> None:
    scores = personalized_pagerank(
        {"a": {}, "b": {"a": 1.0}},
        {"b": 1.0},
        damping=0.85,
        max_iterations=100,
        tolerance=1e-8,
    )

    assert set(scores) == {"a", "b"}
    assert abs(sum(scores.values()) - 1.0) < 1e-6
