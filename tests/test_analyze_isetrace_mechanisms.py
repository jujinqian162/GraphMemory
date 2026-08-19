from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

from scripts.analyze_isetrace_mechanisms import (
    TaskFeature,
    RunMetrics,
    _analyze_comparison,
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


def test_gold_event_bin_collapses_three_or_more_events() -> None:
    def feature(gold_event_count: int) -> TaskFeature:
        return TaskFeature(
            task_id="q1",
            graph_id="g1",
            memory_mode="multi_fact_recall",
            gold_span_count=1,
            gold_event_count=gold_event_count,
            minimum_flat_chunks=1,
            max_provenance_hops=None,
            provenance_connected=None,
        )

    assert feature(1).gold_event_bin == "1"
    assert feature(2).gold_event_bin == "2"
    assert feature(3).gold_event_bin == "3+"


def test_event_dispersion_comparison_uses_seed_matched_task_means() -> None:
    features = {
        "q1": TaskFeature(
            task_id="q1",
            graph_id="g1",
            memory_mode="direct_recall",
            gold_span_count=1,
            gold_event_count=1,
            minimum_flat_chunks=1,
            max_provenance_hops=None,
            provenance_connected=None,
        ),
        "q2": TaskFeature(
            task_id="q2",
            graph_id="g2",
            memory_mode="multi_fact_recall",
            gold_span_count=2,
            gold_event_count=3,
            minimum_flat_chunks=2,
            max_provenance_hops=None,
            provenance_connected=None,
        ),
    }

    def run(seed: int, values: tuple[float, float]) -> RunMetrics:
        return RunMetrics(
            path=Path(f"run-{seed}"),
            method="provenance_rgcn",
            variant="full_rgcn",
            seed=seed,
            test_artifact_digest="test",
            test_payload_digests={},
            metrics={
                "q1": {
                    "Coverage@1024 Tokens": values[0],
                    "Full Support@2048 Tokens": values[0],
                },
                "q2": {
                    "Coverage@1024 Tokens": values[1],
                    "Full Support@2048 Tokens": values[1],
                },
            },
            graph_ids={"q1": "g1", "q2": "g2"},
            memory_modes={
                "q1": "direct_recall",
                "q2": "multi_fact_recall",
            },
        )

    baseline = {seed: run(seed, (0.4, 0.2)) for seed in (13, 17)}
    method = {seed: run(seed, (0.5, 0.5)) for seed in (13, 17)}
    result = _analyze_comparison(
        baseline_runs=baseline,
        method_runs=method,
        features=features,
        bin_name="gold_event_count",
        bin_order=("1", "2", "3+"),
        bin_getter=lambda feature: feature.gold_event_bin,
        bootstrap_samples=100,
        bootstrap_seed=13,
    )

    bins = cast(Sequence[Mapping[str, object]], result["bins"])
    assert [item["bin"] for item in bins] == ["1", "3+"]
    assert bins[0]["task_count"] == 1
    assert bins[1]["task_count"] == 1
    first_metrics = cast(Mapping[str, Mapping[str, object]], bins[0]["metrics"])
    third_metrics = cast(Mapping[str, Mapping[str, object]], bins[1]["metrics"])
    assert (
        abs(
            cast(float, first_metrics["Coverage@1024 Tokens"]["mean_delta"])
            - 0.1
        )
        < 1e-12
    )
    assert (
        abs(
            cast(float, third_metrics["Full Support@2048 Tokens"]["mean_delta"])
            - 0.3
        )
        < 1e-12
    )
    assert (
        first_metrics["Coverage@1024 Tokens"]["paired_query_count"] == 1
    )


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
