import json

from graph_memory.io import merge_config, read_json, write_csv, write_json
from graph_memory.observability import build_run_summary, graph_statistics


def test_json_helpers_write_deterministic_utf8(tmp_path):
    output_path = tmp_path / "artifact.json"
    payload = {"b": "Paris", "a": ["Seine", "Eiffel"]}

    write_json(output_path, payload)

    assert read_json(output_path) == payload
    written = output_path.read_text(encoding="utf-8")
    assert written.startswith('{\n  "a"')
    assert written.endswith("\n")


def test_write_csv_uses_explicit_field_order(tmp_path):
    output_path = tmp_path / "metrics.csv"

    write_csv(output_path, [{"b": "2", "a": "1"}], fieldnames=["a", "b"])

    assert output_path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2"]


def test_merge_config_prefers_cli_over_config_over_defaults():
    defaults = {"seed": 13, "graph": {"max_hops": 1, "enabled": True}}
    config_file = {"graph": {"max_hops": 2}}
    cli_overrides = {"seed": 7}

    assert merge_config(defaults, config_file, cli_overrides) == {
        "seed": 7,
        "graph": {"max_hops": 2, "enabled": True},
    }


def test_build_run_summary_contains_reproducibility_fields():
    summary = build_run_summary(
        script="run_retrieval.py",
        started_at="2026-05-20T12:00:00+08:00",
        finished_at="2026-05-20T12:01:00+08:00",
        status="success",
        effective_config={"method": "bm25"},
        inputs={"tasks": "tasks.input.json"},
        outputs={"predictions": "ranked_results_bm25.json"},
        counts={"tasks": 1},
        timings={"total_seconds": 60.0},
        environment={"python": "3.12"},
        notes=["debug disabled"],
    )

    assert summary["script"] == "run_retrieval.py"
    assert summary["status"] == "success"
    assert summary["effective_config"]["method"] == "bm25"
    assert summary["inputs"]["tasks"] == "tasks.input.json"
    assert summary["outputs"]["predictions"] == "ranked_results_bm25.json"


def test_graph_statistics_counts_edges_and_isolated_memory_nodes():
    graphs = [
        {
            "task_id": "hotpot_ex1",
            "nodes": [
                {"id": "q", "node_type": "question", "text": "query"},
                {"id": "m0", "node_type": "document_sentence", "text": "a", "source": "A", "sentence_id": 0, "position": 0},
                {"id": "m1", "node_type": "document_sentence", "text": "b", "source": "A", "sentence_id": 1, "position": 1},
                {"id": "m2", "node_type": "document_sentence", "text": "c", "source": "B", "sentence_id": 0, "position": 2},
            ],
            "edges": [
                {"source": "m0", "target": "m1", "edge_type": "sequential", "weight": 1.0, "directed": False},
                {"source": "q", "target": "m0", "edge_type": "query_overlap", "weight": 1.0, "directed": True},
            ],
        }
    ]

    stats = graph_statistics(graphs)

    assert stats["num_graphs"] == 1
    assert stats["edge_counts_by_type"] == {"query_overlap": 1, "sequential": 1}
    assert stats["isolated_memory_nodes"] == 1
