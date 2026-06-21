from __future__ import annotations

import json

from graph_memory.datasets.twowiki import (
    TwoWikiToGraphBuildRequest,
    convert_twowiki_example,
    parse_twowiki_example,
)
from graph_memory.graphs.config import GraphBuildConfig
from graph_memory.graphs.construction.builder import build_graphs


def test_twowiki_gold_fields_do_not_enter_input_or_graph_build_request() -> None:
    raw_example = {
        "_id": "abc123",
        "type": "compositional",
        "question": "Who is Ada's mother?",
        "context": [
            ["Film A", ["Film A was directed by Ada."]],
            ["Ada Lovelace", ["Ada was the daughter of Beth."]],
        ],
        "supporting_facts": [["Film A", 0], ["Ada Lovelace", 0]],
        "evidences": [["Film A", "director", "Ada"], ["Ada", "mother", "Beth"]],
        "answer": "Beth",
    }
    converted = convert_twowiki_example(parse_twowiki_example(raw_example))
    graph_request = TwoWikiToGraphBuildRequest().project(converted.ranking_record)

    input_visible_payload = json.dumps(
        {
            "ranking_record": converted.ranking_record,
            "graph_request": {
                "task_id": graph_request.task_id,
                "query_text": graph_request.query_text,
                "nodes": [node.__dict__ for node in graph_request.nodes],
                "input_visible_edges": [edge.__dict__ for edge in graph_request.input_visible_edges],
            },
        },
        sort_keys=True,
    )

    forbidden = [
        "gold_answer",
        "supporting_facts",
        "evidences",
        "evidences_id",
        "answer_id",
        "gold_dependency_edges",
        "is_gold",
    ]
    assert all(field not in input_visible_payload for field in forbidden)


def test_twowiki_gold_fields_do_not_enter_graph_artifacts() -> None:
    raw_example = {
        "_id": "abc123",
        "type": "compositional",
        "question": "Who is Ada's mother?",
        "context": [
            ["Film A", ["Film A was directed by Ada."]],
            ["Ada Lovelace", ["Ada was the daughter of Beth."]],
        ],
        "supporting_facts": [["Film A", 0], ["Ada Lovelace", 0]],
        "evidences": [["Film A", "director", "Ada"], ["Ada", "mother", "Beth"]],
        "answer": "Beth",
    }
    converted = convert_twowiki_example(parse_twowiki_example(raw_example))
    graph_request = TwoWikiToGraphBuildRequest().project(converted.ranking_record)

    graphs = build_graphs(
        [graph_request],
        GraphBuildConfig(max_query_overlap=0, max_entity_neighbors=0, max_bridge_edges=0, use_spacy=False),
    )
    graph_payload = json.dumps(graphs, sort_keys=True)

    forbidden = [
        "gold_answer",
        "supporting_facts",
        "evidences",
        "evidences_id",
        "answer_id",
        "gold_dependency_edges",
        "is_gold",
    ]
    assert all(field not in graph_payload for field in forbidden)
