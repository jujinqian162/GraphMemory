import json

from graph_memory.entities import heuristic_entities, title_aliases
from graph_memory.graphs import build_graph
from graph_memory.text import content_tokens, lexical_score
from graph_memory.types import GraphBuildConfig, MemoryTaskInput


def graph_task_input() -> MemoryTaskInput:
    return {
        "task_id": "hotpot_ex1",
        "query": "Which river runs through the city that hosts the Eiffel Tower?",
        "memory_items": [
            {
                "id": "m0",
                "node_type": "document_sentence",
                "text": "The Eiffel Tower is in Paris.",
                "source": "Eiffel Tower",
                "sentence_id": 0,
                "position": 0,
            },
            {
                "id": "m1",
                "node_type": "document_sentence",
                "text": "It opened in 1889.",
                "source": "Eiffel Tower",
                "sentence_id": 1,
                "position": 1,
            },
            {
                "id": "m2",
                "node_type": "document_sentence",
                "text": "Paris is a city in France.",
                "source": "Paris",
                "sentence_id": 0,
                "position": 2,
            },
            {
                "id": "m3",
                "node_type": "document_sentence",
                "text": "The Seine runs through Paris.",
                "source": "Paris",
                "sentence_id": 1,
                "position": 3,
            },
        ],
    }


def test_content_tokens_drop_stopwords_and_keep_entities():
    tokens = content_tokens("Which city hosts the Eiffel Tower and what river runs through it?")

    assert "and" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "eiffel" in tokens
    assert "tower" in tokens
    assert "river" in tokens


def test_lexical_score_rewards_content_overlap_more_than_stopwords():
    idf = {"eiffel": 3.0, "tower": 3.0, "the": 0.0, "and": 0.0}

    assert lexical_score("the Eiffel Tower", "Eiffel Tower is in Paris", idf) > lexical_score(
        "the and", "the and", idf
    )


def test_heuristic_entities_and_title_aliases_are_deterministic():
    assert "eiffel tower" in heuristic_entities("The Eiffel Tower is in Paris.")
    assert title_aliases("Eiffel Tower") == {"eiffel tower", "eiffel", "tower"}


def test_graph_builds_typed_edges_without_label_fields():
    config = GraphBuildConfig(max_query_overlap=20, max_entity_neighbors=10, max_bridge_edges=50)
    graph = build_graph(graph_task_input(), config)
    encoded = json.dumps(graph)
    edge_types = {edge["edge_type"] for edge in graph["edges"]}

    assert "gold_answer" not in encoded
    assert "gold_evidence_nodes" not in encoded
    assert {"sequential", "query_overlap", "entity_overlap", "bridge"}.issubset(edge_types)
    assert sum(1 for node in graph["nodes"] if node["id"] == "q") == 1
    assert {node["id"] for node in graph["nodes"]} == {"q", "m0", "m1", "m2", "m3"}
    assert any(edge["source"] == "q" and edge["edge_type"] == "query_overlap" for edge in graph["edges"])


def test_graph_respects_query_overlap_limit():
    config = GraphBuildConfig(max_query_overlap=1, max_entity_neighbors=10, max_bridge_edges=50)
    graph = build_graph(graph_task_input(), config)

    assert sum(1 for edge in graph["edges"] if edge["edge_type"] == "query_overlap") == 1
