from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.analyze_isetrace_mechanisms import (
    iter_json_array,
    main,
    max_observable_provenance_distance,
    minimum_covering_candidates,
)


def _span(event: str, start: int, end: int) -> dict[str, object]:
    return {
        "event_id": event,
        "json_pointer": "/content",
        "char_start": start,
        "char_end": end,
    }


def _candidate(*spans: dict[str, object]) -> dict[str, object]:
    return {"source_spans": list(spans)}


def test_iter_json_array_streams_across_small_chunks(tmp_path: Path) -> None:
    path = tmp_path / "values.json"
    expected = [{"text": "x" * 31}, {"value": 2}, [1, 2, 3]]
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert list(iter_json_array(path, chunk_size=7)) == expected


def test_minimum_covering_candidates_solves_partial_span_set_cover() -> None:
    gold = [_span("e1", 0, 10), _span("e2", 0, 3)]
    candidates = [
        _candidate(_span("e1", 0, 6)),
        _candidate(_span("e1", 4, 10), _span("e2", 0, 3)),
        _candidate(_span("e1", 2, 5)),
    ]

    assert minimum_covering_candidates(gold, candidates) == 2
    assert (
        minimum_covering_candidates(
            gold,
            [_candidate(_span("e1", 0, 10), _span("e2", 0, 3))],
        )
        == 1
    )


def test_provenance_distance_excludes_temporal_edges_and_detects_disconnect() -> None:
    graph = {
        "graph_id": "g1",
        "nodes": [
            {
                "node_id": "o1",
                "kind": "execution.tool_output",
                "source_spans": [{"event_id": "e1"}],
            },
            {
                "node_id": "c1",
                "kind": "content.tool_output",
                "source_spans": [{"event_id": "e1", "json_pointer": "/content"}],
            },
            {
                "node_id": "c2",
                "kind": "content.tool_argument",
                "source_spans": [{"event_id": "e2", "json_pointer": "/content"}],
            },
            {
                "node_id": "o2",
                "kind": "execution.tool_call",
                "source_spans": [{"event_id": "e2"}],
            },
            {
                "node_id": "o3",
                "kind": "execution.tool_output",
                "source_spans": [{"event_id": "e3"}],
            },
        ],
        "edges": [
            {"source": "o1", "target": "o2", "relation": "temporal.precedes"},
            {"source": "o1", "target": "c1", "relation": "execution.has_content"},
            {"source": "c1", "target": "c2", "relation": "data.feeds"},
            {"source": "o2", "target": "c2", "relation": "execution.has_argument"},
        ],
    }

    assert max_observable_provenance_distance(graph, ["e1", "e2"]) == (3, True)
    assert max_observable_provenance_distance(graph, ["e1", "e3"]) == (None, False)
    assert max_observable_provenance_distance(graph, ["e1"]) == (None, None)


def _write_run(
    root: Path,
    *,
    method: str,
    variant: str,
    seed: int,
    coverage: tuple[float, float],
    support: tuple[float, float],
) -> None:
    (root / "workflow").mkdir(parents=True)
    (root / "metrics").mkdir()
    (root / "assets").mkdir()
    (root / "workflow" / "summary.yaml").write_text(
        yaml.safe_dump(
            {
                "method": method,
                "variant": variant,
                "dataset": "isetrace",
                "profile": "full",
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )
    (root / "assets" / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "assets": [
                    {
                        "kind": "dataset",
                        "digest": "fixed-test",
                        "origin": {"split": "test"},
                        "payloads": [
                            {"role": role, "digest": f"{role}-digest"}
                            for role in ("labels", "tasks", "provenance_graphs")
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "q1",
            "graph_id": "g1",
            "memory_mode": "direct_recall",
            "Coverage@1024 Tokens": coverage[0],
            "Full Support@2048 Tokens": support[0],
        },
        {
            "task_id": "q2",
            "graph_id": "g2",
            "memory_mode": "multi_fact_recall",
            "Coverage@1024 Tokens": coverage[1],
            "Full Support@2048 Tokens": support[1],
        },
    ]
    with (root / "metrics" / "per_task.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _write_prepared(root: Path) -> None:
    root.mkdir()
    labels = [
        {
            "task_id": "q1",
            "graph_id": "g1",
            "gold_evidence_spans": [_span("e1", 0, 4)],
        },
        {
            "task_id": "q2",
            "graph_id": "g2",
            "gold_evidence_spans": [_span("e2", 0, 4), _span("e3", 0, 4)],
        },
    ]
    tasks = [
        {
            "task_id": "q1",
            "graph_id": "g1",
            "flat_candidates": [_candidate(_span("e1", 0, 4))],
        },
        {
            "task_id": "q2",
            "graph_id": "g2",
            "flat_candidates": [
                _candidate(_span("e2", 0, 4)),
                _candidate(_span("e3", 0, 4)),
            ],
        },
    ]
    graphs = [
        {
            "graph_id": "g2",
            "nodes": [
                {
                    "node_id": "n2",
                    "kind": "execution.tool_output",
                    "source_spans": [{"event_id": "e2"}],
                },
                {
                    "node_id": "n3",
                    "kind": "execution.tool_call",
                    "source_spans": [{"event_id": "e3"}],
                },
            ],
            "edges": [
                {"source": "n2", "target": "n3", "relation": "execution.returns"}
            ],
        }
    ]
    (root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (root / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (root / "provenance_graphs.json").write_text(json.dumps(graphs), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "digest": "prepared-test",
                "origin": {"source_revision": "revision"},
                "payloads": [
                    {"role": role, "digest": f"{role}-digest"}
                    for role in ("labels", "tasks", "provenance_graphs")
                ],
            }
        ),
        encoding="utf-8",
    )


def test_main_builds_fragmentation_and_provenance_mechanism_results(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    _write_prepared(prepared)
    groups: dict[str, list[Path]] = {
        "flat": [],
        "pu": [],
        "seed": [],
        "rgcn": [],
    }
    for seed in (13, 17):
        for group, method, variant, coverage, support in (
            ("flat", "dense_ft", "flat", (0.5, 0.4), (0.4, 0.2)),
            ("pu", "dense_ft", "provenance_unit", (0.6, 0.7), (0.5, 0.6)),
            ("seed", "provenance_rgcn", "wo_graph", (0.6, 0.7), (0.5, 0.6)),
            ("rgcn", "provenance_rgcn", "full_rgcn", (0.6, 0.8), (0.5, 0.8)),
        ):
            run = tmp_path / f"{group}-{seed}"
            _write_run(
                run,
                method=method,
                variant=variant,
                seed=seed,
                coverage=coverage,
                support=support,
            )
            groups[group].append(run)

    output = tmp_path / "mechanisms.json"
    output_csv = tmp_path / "mechanisms.csv"
    argv = [
        "--prepared-dir",
        str(prepared),
        "--output",
        str(output),
        "--output-csv",
        str(output_csv),
        "--bootstrap-samples",
        "100",
    ]
    for option, group in (
        ("--flat-run", "flat"),
        ("--provenance-unit-run", "pu"),
        ("--seed-run", "seed"),
        ("--rgcn-run", "rgcn"),
    ):
        for run in groups[group]:
            argv.extend((option, str(run)))

    assert main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["test_task_count"] == 2
    assert result["feature_counts"]["fragmentation_bins"] == {"1": 1, "2": 1}
    assert result["feature_counts"]["provenance_bins"] == {"1": 1}
    segmentation_bins = result["comparisons"]["segmentation"]["bins"]
    assert [row["bin"] for row in segmentation_bins] == ["1", "2"]
    assert segmentation_bins[1]["metrics"]["Coverage@1024 Tokens"][
        "mean_delta"
    ] == pytest.approx(0.3)
    graph_bin = result["comparisons"]["graph_residual"]["bins"][0]
    assert graph_bin["bin"] == "1"
    assert graph_bin["metrics"]["Full Support@2048 Tokens"][
        "mean_delta"
    ] == pytest.approx(0.2)
    assert output_csv.is_file()
