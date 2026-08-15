from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.analyze_isetrace_mechanisms import (
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
