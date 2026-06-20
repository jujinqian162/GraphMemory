from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from pathlib import Path

from graph_memory.datasets.hotpotqa.projectors import HotpotQAToGraphBuildRequest
from graph_memory.datasets.hotpotqa.records import HotpotQARankingRecord


ROOT = Path(__file__).resolve().parents[1]
GRAPH_BUILD_CONFIG_OWNER = "graph_memory.graphs.config"


def _task_input() -> HotpotQARankingRecord:
    return {
        "task_id": "hotpot_graph_batch3",
        "question": "Which river runs through the city that hosts the Eiffel Tower?",
        "candidate_sentences": [
            {
                "sentence_id": "m0",
                "text": "The Eiffel Tower is in Paris.",
                "title": "Eiffel Tower",
                "sentence_index": 0,
                "position": 0,
            },
            {
                "sentence_id": "m1",
                "text": "It opened in 1889.",
                "title": "Eiffel Tower",
                "sentence_index": 1,
                "position": 1,
            },
            {
                "sentence_id": "m2",
                "text": "Paris is a city in France.",
                "title": "Paris",
                "sentence_index": 0,
                "position": 2,
            },
            {
                "sentence_id": "m3",
                "text": "The Seine runs through Paris.",
                "title": "Paris",
                "sentence_index": 1,
                "position": 3,
            },
        ],
    }


def _imported_graph_build_config_from_types(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "graph_memory.types":
            continue
        imported_names = {alias.name for alias in node.names}
        if "GraphBuildConfig" in imported_names:
            return True
    return False


def _assert_edge(actual: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for key, value in expected.items():
        if key == "weight":
            actual_weight = actual[key]
            assert isinstance(actual_weight, (int, float))
            assert isinstance(value, (int, float))
            assert math.isclose(float(actual_weight), float(value), rel_tol=1e-12, abs_tol=1e-12)
        else:
            assert actual[key] == value


def test_graph_domain_package_exposes_config_builder_index_statistics_and_views() -> None:
    import graph_memory.graphs as graphs_package
    import graph_memory.graphs.construction as construction_package
    from graph_memory.graphs.config import GraphBuildConfig
    from graph_memory.graphs.construction.builder import GraphBuilder, build_graphs
    from graph_memory.graphs.construction.rules.bridge import BridgeEdgeRule
    from graph_memory.graphs.construction.rules.entity_overlap import EntityOverlapEdgeRule
    from graph_memory.graphs.construction.rules.query_overlap import QueryOverlapEdgeRule
    from graph_memory.graphs.construction.rules.sequential import SequentialEdgeRule
    from graph_memory.graphs.index import GraphIndex
    from graph_memory.graphs.statistics import graph_statistics
    from graph_memory.graphs.views import induced_retrieved_subgraph, model_visible_graph, traversal_adjacency

    assert (ROOT / "graph_memory" / "graphs" / "__init__.py").exists()
    assert not (ROOT / "graph_memory" / "graphs.py").exists()
    assert not hasattr(graphs_package, "build_graph")
    assert not hasattr(construction_package, "build_graph")
    assert GraphBuildConfig.__module__ == GRAPH_BUILD_CONFIG_OWNER

    builder = GraphBuilder(GraphBuildConfig(max_query_overlap=20, max_entity_neighbors=10, max_bridge_edges=50))
    assert tuple(type(rule) for rule in builder.rules) == (
        SequentialEdgeRule,
        QueryOverlapEdgeRule,
        EntityOverlapEdgeRule,
        BridgeEdgeRule,
    )

    request = HotpotQAToGraphBuildRequest().project(_task_input())
    graph = builder.build(request)
    assert build_graphs([request], builder.config) == [graph]
    assert GraphIndex.from_graphs([graph]).graph_by_task_id == {"hotpot_graph_batch3": graph}

    expected_edges = [
        {"source": "m0", "target": "m1", "edge_type": "sequential", "weight": 1.0, "directed": False},
        {"source": "m2", "target": "m3", "edge_type": "sequential", "weight": 1.0, "directed": False},
        {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 9.31093021621633, "directed": True},
        {"source": "q", "target": "m1", "edge_type": "query_overlap", "weight": 9.31093021621633, "directed": True},
        {"source": "q", "target": "m2", "edge_type": "query_overlap", "weight": 1.6931471805599454, "directed": True},
        {"source": "q", "target": "m3", "edge_type": "query_overlap", "weight": 1.6931471805599454, "directed": True},
        {"source": "m0", "target": "m1", "edge_type": "entity_overlap", "weight": 3.0, "directed": False},
        {"source": "m0", "target": "m2", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        {"source": "m0", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        {"source": "m2", "target": "m3", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
        {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 2.0, "directed": False},
        {"source": "m0", "target": "m3", "edge_type": "bridge", "weight": 2.0, "directed": False},
    ]
    assert len(graph["edges"]) == len(expected_edges)
    for actual, expected in zip(graph["edges"], expected_edges):
        _assert_edge(actual, expected)

    stats = graph_statistics([graph], split="dev", graph_config={"max_query_overlap": 20})
    assert stats == {
        "num_graphs": 1,
        "avg_nodes": 5.0,
        "avg_edges": 12.0,
        "edge_counts_by_type": {
            "bridge": 2,
            "entity_overlap": 4,
            "query_overlap": 4,
            "sequential": 2,
        },
        "isolated_memory_nodes": 0,
        "split": "dev",
        "graph_config": {"max_query_overlap": 20},
    }
    assert induced_retrieved_subgraph(graph, ["m0", "m2"]) == {
        "nodes": ["m0", "m2"],
        "edges": [
            {"source": "m0", "target": "m2", "edge_type": "entity_overlap", "weight": 1.0, "directed": False},
            {"source": "m0", "target": "m2", "edge_type": "bridge", "weight": 2.0, "directed": False},
        ],
    }
    assert traversal_adjacency(graph)["m2"] == {"m0", "m3"}
    assert [edge["edge_type"] for edge in model_visible_graph(graph, frozenset({"bridge"}))["edges"]] == [
        "bridge",
        "bridge",
    ]


def test_edge_accumulator_preserves_directed_keys_and_undirected_deduplication() -> None:
    from graph_memory.graphs.construction.edge_accumulator import EdgeAccumulator

    accumulator = EdgeAccumulator()
    accumulator.add("m0", "m1", "sequential", 1.0, directed=False)
    accumulator.add("m1", "m0", "sequential", 2.0, directed=False)
    accumulator.add("m0", "m1", "query_overlap", 3.0, directed=True)
    accumulator.add("m1", "m0", "query_overlap", 4.0, directed=True)

    assert accumulator.edges == [
        {"source": "m0", "target": "m1", "edge_type": "sequential", "weight": 1.0, "directed": False},
        {"source": "m0", "target": "m1", "edge_type": "query_overlap", "weight": 3.0, "directed": True},
        {"source": "m1", "target": "m0", "edge_type": "query_overlap", "weight": 4.0, "directed": True},
    ]


def test_graph_build_config_imports_are_migrated_from_types() -> None:
    scanned_roots = [ROOT / "graph_memory", ROOT / "scripts", ROOT / "tests"]
    offenders: list[str] = []
    for scanned_root in scanned_roots:
        for path in scanned_root.rglob("*.py"):
            if path == ROOT / "graph_memory" / "types.py":
                continue
            if _imported_graph_build_config_from_types(path):
                offenders.append(str(path.relative_to(ROOT)))

    assert sorted(offenders) == []
