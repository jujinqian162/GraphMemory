from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.analysis import paired_cluster_delta_bootstrap_ci

MECHANISM_METRICS = (
    "Coverage@1024 Tokens",
    "Full Support@2048 Tokens",
)
TEMPORAL_RELATION = "temporal.precedes"
OWNER_NODE_KINDS = {"execution.tool_call", "execution.tool_output"}
FRAGMENTATION_BIN_ORDER = ("1", "2", "3+")
GOLD_EVENT_BIN_ORDER = ("1", "2", "3+")
PROVENANCE_BIN_ORDER = ("1", "2", "3+", "disconnected")


@dataclass(frozen=True)
class SourceInterval:
    event_id: str
    json_pointer: str
    start: int
    end: int


@dataclass(frozen=True)
class TaskFeature:
    task_id: str
    graph_id: str
    memory_mode: str
    gold_span_count: int
    gold_event_count: int
    minimum_flat_chunks: int
    max_provenance_hops: int | None
    provenance_connected: bool | None

    @property
    def fragmentation_bin(self) -> str:
        return "3+" if self.minimum_flat_chunks >= 3 else str(self.minimum_flat_chunks)

    @property
    def gold_event_bin(self) -> str:
        return "3+" if self.gold_event_count >= 3 else str(self.gold_event_count)

    @property
    def provenance_bin(self) -> str | None:
        if self.gold_event_count < 2:
            return None
        if self.provenance_connected is False:
            return "disconnected"
        if self.max_provenance_hops is None:
            raise ValueError(f"task_id={self.task_id!r} lacks provenance distance")
        return "3+" if self.max_provenance_hops >= 3 else str(self.max_provenance_hops)


@dataclass(frozen=True)
class RunMetrics:
    path: Path
    method: str
    variant: str | None
    seed: int
    test_artifact_digest: str
    test_payload_digests: Mapping[str, str]
    metrics: Mapping[str, Mapping[str, float]]
    graph_ids: Mapping[str, str]
    memory_modes: Mapping[str, str]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ISETrace segmentation and graph-residual mechanisms from "
            "frozen prepared data and completed per-task results."
        )
    )
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--flat-run", type=Path, action="append", required=True)
    parser.add_argument(
        "--provenance-unit-run", type=Path, action="append", required=True
    )
    parser.add_argument("--seed-run", type=Path, action="append", required=True)
    parser.add_argument("--rgcn-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--dispersion-figure", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args(argv)

    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    flat_runs = _load_run_group(args.flat_run)
    provenance_unit_runs = _load_run_group(args.provenance_unit_run)
    seed_runs = _load_run_group(args.seed_run)
    rgcn_runs = _load_run_group(args.rgcn_run)
    _validate_seed_pairs(flat_runs, provenance_unit_runs, comparison="segmentation")
    _validate_seed_pairs(seed_runs, rgcn_runs, comparison="graph residual")

    all_runs = [
        *flat_runs.values(),
        *provenance_unit_runs.values(),
        *seed_runs.values(),
        *rgcn_runs.values(),
    ]
    canonical_run = all_runs[0]
    _validate_run_alignment(all_runs)

    prepared = _load_prepared_inputs(args.prepared_dir)
    prepared_identity = _mapping(prepared["identity"], "prepared identity")
    prepared_payloads = _mapping(
        prepared_identity["payload_digests"], "prepared identity.payload_digests"
    )
    if canonical_run.test_payload_digests != prepared_payloads:
        raise ValueError("prepared payload digests do not match the completed runs")
    task_features = _compute_task_features(
        prepared_dir=args.prepared_dir,
        labels=cast(Mapping[str, Mapping[str, object]], prepared["labels"]),
        memory_modes=canonical_run.memory_modes,
    )
    expected_tasks = set(canonical_run.metrics)
    if set(task_features) != expected_tasks:
        missing = sorted(expected_tasks - set(task_features))
        extra = sorted(set(task_features) - expected_tasks)
        raise ValueError(
            f"prepared/run task mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    comparisons = {
        "segmentation": _analyze_comparison(
            baseline_runs=flat_runs,
            method_runs=provenance_unit_runs,
            features=task_features,
            bin_name="minimum_flat_chunks",
            bin_order=FRAGMENTATION_BIN_ORDER,
            bin_getter=lambda feature: feature.fragmentation_bin,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "graph_residual": _analyze_comparison(
            baseline_runs=seed_runs,
            method_runs=rgcn_runs,
            features=task_features,
            bin_name="max_observable_provenance_hops",
            bin_order=PROVENANCE_BIN_ORDER,
            bin_getter=lambda feature: feature.provenance_bin,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "gold_event_dispersion": _analyze_comparison(
            baseline_runs=seed_runs,
            method_runs=rgcn_runs,
            features=task_features,
            bin_name="gold_event_count",
            bin_order=GOLD_EVENT_BIN_ORDER,
            bin_getter=lambda feature: feature.gold_event_bin,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    }

    result: dict[str, object] = {
        "schema_version": 1,
        "dataset": "isetrace",
        "test_task_count": len(task_features),
        "test_trajectory_count": len(
            {item.graph_id for item in task_features.values()}
        ),
        "metrics": list(MECHANISM_METRICS),
        "bootstrap": {
            "unit": "trajectory",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "seed_pair_reduction": "mean per task before cluster resampling",
        },
        "prepared_artifact": prepared_identity,
        "run_test_artifact_digest": canonical_run.test_artifact_digest,
        "definitions": {
            "minimum_flat_chunks": (
                "Exact minimum number of frozen flat candidates whose source-span "
                "union covers every gold source character."
            ),
            "max_observable_provenance_hops": (
                "Maximum undirected shortest-path distance between distinct "
                "gold-bearing tool-call/tool-output event nodes after excluding "
                "temporal.precedes; undefined for one-event queries."
            ),
            "gold_event_count": (
                "Number of distinct event IDs containing at least one gold "
                "evidence span; bins are 1, 2, and 3+."
            ),
        },
        "feature_counts": _feature_counts(task_features.values()),
        "comparisons": comparisons,
        "task_features": [
            {
                "task_id": item.task_id,
                "graph_id": item.graph_id,
                "memory_mode": item.memory_mode,
                "gold_span_count": item.gold_span_count,
                "gold_event_count": item.gold_event_count,
                "minimum_flat_chunks": item.minimum_flat_chunks,
                "fragmentation_bin": item.fragmentation_bin,
                "max_provenance_hops": item.max_provenance_hops,
                "provenance_connected": item.provenance_connected,
                "provenance_bin": item.provenance_bin,
            }
            for item in sorted(task_features.values(), key=lambda value: value.task_id)
        ],
    }
    _write_json(args.output, result)
    _write_summary_csv(args.output_csv, comparisons)
    if args.figure is not None:
        _write_figure(args.figure, comparisons)
    if args.dispersion_figure is not None:
        _write_dispersion_figure(
            args.dispersion_figure,
            _mapping(comparisons["gold_event_dispersion"], "gold_event_dispersion"),
        )
    return 0


def iter_json_array(
    path: Path, *, chunk_size: int = 8 * 1024 * 1024
) -> Iterator[object]:
    """Stream top-level JSON-array values without loading the complete artifact."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        eof = False
        started = False
        expect_value = True
        while True:
            if not eof and (len(buffer) < chunk_size or not buffer.strip()):
                piece = stream.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True

            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        raise ValueError(f"empty JSON array artifact: {path}")
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"expected top-level JSON array: {path}")
                buffer = buffer[1:]
                started = True
                continue

            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    raise ValueError(f"unterminated JSON array: {path}")
                continue
            if buffer[0] == "]":
                if buffer[1:].strip():
                    raise ValueError(f"trailing content after JSON array: {path}")
                return
            if not expect_value:
                if buffer[0] != ",":
                    raise ValueError(f"expected comma in JSON array: {path}")
                buffer = buffer[1:]
                expect_value = True
                continue

            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f"invalid JSON array artifact: {path}") from error
                piece = stream.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True
                continue
            yield value
            buffer = buffer[end:]
            expect_value = False


def minimum_covering_candidates(
    gold_spans: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
) -> int:
    """Return the exact set-cover size over atomic gold source intervals."""
    gold = _source_intervals(gold_spans)
    if not gold:
        raise ValueError("gold spans must be non-empty")
    candidate_intervals = [
        _source_intervals(_mapping_list(candidate.get("source_spans"), "source_spans"))
        for candidate in candidates
    ]
    atoms = _gold_atoms(gold, candidate_intervals)
    full_mask = (1 << len(atoms)) - 1
    masks = {
        _coverage_mask(intervals, atoms)
        for intervals in candidate_intervals
        if intervals
    }
    masks.discard(0)
    if not masks:
        raise ValueError("flat candidates do not overlap any gold source span")
    if full_mask not in _reachable_unions(masks):
        raise ValueError("flat candidates do not fully cover the gold source spans")

    undominated = {
        mask
        for mask in masks
        if not any(mask != other and mask | other == other for other in masks)
    }
    states = {0}
    for depth in range(1, len(undominated) + 1):
        next_states = {
            state | mask
            for state in states
            for mask in undominated
            if state | mask != state
        }
        if full_mask in next_states:
            return depth
        states = {
            state
            for state in next_states
            if not any(
                state != other and state | other == other for other in next_states
            )
        }
    raise ValueError("could not solve flat-candidate gold coverage")


def max_observable_provenance_distance(
    graph: Mapping[str, object],
    gold_event_ids: Sequence[str],
) -> tuple[int | None, bool | None]:
    unique_events = tuple(sorted(set(gold_event_ids)))
    if len(unique_events) < 2:
        return None, None

    nodes = _mapping_list(graph.get("nodes"), "graph.nodes")
    edges = _mapping_list(graph.get("edges"), "graph.edges")
    owner_by_event: dict[str, str] = {}
    adjacency: dict[str, set[str]] = {}
    for node in nodes:
        node_id = _required_string(node.get("node_id"), "graph.nodes.node_id")
        adjacency.setdefault(node_id, set())
        if node.get("kind") not in OWNER_NODE_KINDS:
            continue
        for span in _mapping_list(node.get("source_spans"), "node.source_spans"):
            event_id = _required_string(
                span.get("event_id"), "node.source_spans.event_id"
            )
            previous = owner_by_event.setdefault(event_id, node_id)
            if previous != node_id:
                raise ValueError(f"event_id={event_id!r} has multiple owner nodes")
    for edge in edges:
        relation = _required_string(edge.get("relation"), "graph.edges.relation")
        if relation == TEMPORAL_RELATION:
            continue
        source = _required_string(edge.get("source"), "graph.edges.source")
        target = _required_string(edge.get("target"), "graph.edges.target")
        if source not in adjacency or target not in adjacency:
            raise ValueError(f"edge references unknown node: {source!r}->{target!r}")
        adjacency[source].add(target)
        adjacency[target].add(source)

    missing = [event_id for event_id in unique_events if event_id not in owner_by_event]
    if missing:
        graph_id = graph.get("graph_id")
        raise ValueError(
            f"graph_id={graph_id!r} lacks owner nodes for events={missing}"
        )
    owners = [owner_by_event[event_id] for event_id in unique_events]
    maximum = 0
    for index, source in enumerate(owners[:-1]):
        distances = _shortest_distances(adjacency, source)
        for target in owners[index + 1 :]:
            distance = distances.get(target)
            if distance is None:
                return None, False
            maximum = max(maximum, distance)
    return maximum, True


def _compute_task_features(
    *,
    prepared_dir: Path,
    labels: Mapping[str, Mapping[str, object]],
    memory_modes: Mapping[str, str],
) -> dict[str, TaskFeature]:
    partial: dict[str, TaskFeature] = {}
    tasks_path = prepared_dir / "tasks.json"
    for raw in iter_json_array(tasks_path):
        task = _mapping(raw, "task")
        task_id = _required_string(task.get("task_id"), "task.task_id")
        graph_id = _required_string(task.get("graph_id"), "task.graph_id")
        label = labels.get(task_id)
        if label is None:
            raise ValueError(f"task_id={task_id!r} has no label")
        label_graph_id = _required_string(label.get("graph_id"), "label.graph_id")
        if label_graph_id != graph_id:
            raise ValueError(f"task_id={task_id!r} has inconsistent graph IDs")
        gold_spans = _mapping_list(
            label.get("gold_evidence_spans"), "gold_evidence_spans"
        )
        flat_candidates = _mapping_list(task.get("flat_candidates"), "flat_candidates")
        gold_events = {
            _required_string(span.get("event_id"), "gold_evidence_spans.event_id")
            for span in gold_spans
        }
        mode = memory_modes.get(task_id)
        if mode is None:
            raise ValueError(
                f"task_id={task_id!r} lacks memory_mode in per-task results"
            )
        if task_id in partial:
            raise ValueError(f"duplicate task_id={task_id!r} in {tasks_path}")
        partial[task_id] = TaskFeature(
            task_id=task_id,
            graph_id=graph_id,
            memory_mode=mode,
            gold_span_count=len(gold_spans),
            gold_event_count=len(gold_events),
            minimum_flat_chunks=minimum_covering_candidates(
                gold_spans, flat_candidates
            ),
            max_provenance_hops=None,
            provenance_connected=None,
        )
    if set(partial) != set(labels):
        missing = sorted(set(labels) - set(partial))
        raise ValueError(f"tasks do not cover all labels; missing={missing[:5]}")

    tasks_by_graph: dict[str, list[TaskFeature]] = {}
    for feature in partial.values():
        if feature.gold_event_count >= 2:
            tasks_by_graph.setdefault(feature.graph_id, []).append(feature)
    observed_graphs: set[str] = set()
    completed = dict(partial)
    for raw in iter_json_array(prepared_dir / "provenance_graphs.json"):
        graph = _mapping(raw, "provenance graph")
        graph_id = _required_string(graph.get("graph_id"), "graph.graph_id")
        graph_tasks = tasks_by_graph.get(graph_id)
        if not graph_tasks:
            continue
        observed_graphs.add(graph_id)
        for feature in graph_tasks:
            label = labels[feature.task_id]
            gold_events = [
                _required_string(span.get("event_id"), "gold_evidence_spans.event_id")
                for span in _mapping_list(
                    label.get("gold_evidence_spans"), "gold_evidence_spans"
                )
            ]
            hops, connected = max_observable_provenance_distance(graph, gold_events)
            completed[feature.task_id] = TaskFeature(
                task_id=feature.task_id,
                graph_id=feature.graph_id,
                memory_mode=feature.memory_mode,
                gold_span_count=feature.gold_span_count,
                gold_event_count=feature.gold_event_count,
                minimum_flat_chunks=feature.minimum_flat_chunks,
                max_provenance_hops=hops,
                provenance_connected=connected,
            )
    missing_graphs = sorted(set(tasks_by_graph) - observed_graphs)
    if missing_graphs:
        raise ValueError(
            f"prepared graphs missing required graph IDs={missing_graphs[:5]}"
        )
    return completed


def _analyze_comparison(
    *,
    baseline_runs: Mapping[int, RunMetrics],
    method_runs: Mapping[int, RunMetrics],
    features: Mapping[str, TaskFeature],
    bin_name: str,
    bin_order: Sequence[str],
    bin_getter: Any,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    seeds = sorted(baseline_runs)
    task_ids = sorted(next(iter(baseline_runs.values())).metrics)
    bins: list[dict[str, object]] = []
    for bin_value in bin_order:
        selected = [
            task_id
            for task_id in task_ids
            if bin_getter(features[task_id]) == bin_value
        ]
        if not selected:
            continue
        metric_results: dict[str, object] = {}
        for metric in MECHANISM_METRICS:
            baseline_by_task: dict[str, float] = {}
            method_by_task: dict[str, float] = {}
            delta_by_cluster: dict[str, list[float]] = {}
            for task_id in selected:
                baseline_values = [
                    baseline_runs[seed].metrics[task_id][metric] for seed in seeds
                ]
                method_values = [
                    method_runs[seed].metrics[task_id][metric] for seed in seeds
                ]
                baseline_mean = sum(baseline_values) / len(baseline_values)
                method_mean = sum(method_values) / len(method_values)
                baseline_by_task[task_id] = baseline_mean
                method_by_task[task_id] = method_mean
                delta_by_cluster.setdefault(features[task_id].graph_id, []).append(
                    method_mean - baseline_mean
                )
            bootstrap = paired_cluster_delta_bootstrap_ci(
                delta_by_cluster,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            metric_results[metric] = {
                "baseline_mean": sum(baseline_by_task.values()) / len(selected),
                "method_mean": sum(method_by_task.values()) / len(selected),
                **bootstrap,
            }
        bins.append(
            {
                "bin": bin_value,
                "task_count": len(selected),
                "trajectory_count": len(
                    {features[task_id].graph_id for task_id in selected}
                ),
                "memory_mode_counts": _counts(
                    features[task_id].memory_mode for task_id in selected
                ),
                "metrics": metric_results,
            }
        )
    return {
        "bin_name": bin_name,
        "delta_direction": "method_minus_baseline",
        "baseline": _run_group_identity(baseline_runs),
        "method": _run_group_identity(method_runs),
        "bins": bins,
    }


def _load_prepared_inputs(prepared_dir: Path) -> dict[str, object]:
    manifest = _mapping(_read_json(prepared_dir / "manifest.json"), "manifest")
    labels_raw = _read_json(prepared_dir / "labels.json")
    if not isinstance(labels_raw, list):
        raise ValueError("labels.json must contain a JSON array")
    labels: dict[str, Mapping[str, object]] = {}
    for raw in labels_raw:
        label = _mapping(raw, "label")
        task_id = _required_string(label.get("task_id"), "label.task_id")
        if task_id in labels:
            raise ValueError(f"duplicate task_id={task_id!r} in labels")
        labels[task_id] = label
    payloads = _mapping_list(manifest.get("payloads"), "manifest.payloads")
    payload_digests = {
        _required_string(payload.get("role"), "payload.role"): _required_string(
            payload.get("digest"), "payload.digest"
        )
        for payload in payloads
    }
    for role in ("labels", "tasks", "provenance_graphs"):
        if role not in payload_digests:
            raise ValueError(f"manifest lacks required payload role={role!r}")
    return {
        "labels": labels,
        "identity": {
            "manifest_digest": manifest.get("digest"),
            "payload_digests": {
                role: payload_digests[role]
                for role in ("labels", "tasks", "provenance_graphs")
            },
            "source_revision": _mapping(manifest.get("origin"), "manifest.origin").get(
                "source_revision"
            ),
        },
    }


def _load_run_group(paths: Sequence[Path]) -> dict[int, RunMetrics]:
    runs: dict[int, RunMetrics] = {}
    for path in paths:
        run = _load_run(path)
        if run.seed in runs:
            raise ValueError(f"duplicate seed={run.seed} in run group")
        runs[run.seed] = run
    if not runs:
        raise ValueError("run group must be non-empty")
    return runs


def _load_run(path: Path) -> RunMetrics:
    summary = _mapping(_read_yaml(path / "workflow" / "summary.yaml"), "run summary")
    method = _required_string(summary.get("method"), "summary.method")
    variant_value = summary.get("variant")
    if variant_value is not None and not isinstance(variant_value, str):
        raise ValueError(f"invalid variant in {path}")
    seed = summary.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"invalid seed in {path}")
    metrics: dict[str, dict[str, float]] = {}
    graph_ids: dict[str, str] = {}
    memory_modes: dict[str, str] = {}
    with (path / "metrics" / "per_task.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = _mapping(raw, f"{path}/per_task.jsonl:{line_number}")
            task_id = _required_string(row.get("task_id"), "per-task.task_id")
            if task_id in metrics:
                raise ValueError(f"duplicate task_id={task_id!r} in {path}")
            metrics[task_id] = {
                metric: _required_number(row.get(metric), f"per-task.{metric}")
                for metric in MECHANISM_METRICS
            }
            graph_ids[task_id] = _required_string(
                row.get("graph_id"), "per-task.graph_id"
            )
            memory_modes[task_id] = _required_string(
                row.get("memory_mode"), "per-task.memory_mode"
            )
    if not metrics:
        raise ValueError(f"run has no per-task metrics: {path}")
    test_artifact_digest, test_payload_digests = _run_test_artifact_identity(path)
    return RunMetrics(
        path=path,
        method=method,
        variant=cast(str | None, variant_value),
        seed=seed,
        test_artifact_digest=test_artifact_digest,
        test_payload_digests=test_payload_digests,
        metrics=metrics,
        graph_ids=graph_ids,
        memory_modes=memory_modes,
    )


def _validate_seed_pairs(
    baseline: Mapping[int, RunMetrics],
    method: Mapping[int, RunMetrics],
    *,
    comparison: str,
) -> None:
    if set(baseline) != set(method):
        raise ValueError(
            f"{comparison} seeds differ: baseline={sorted(baseline)}, "
            f"method={sorted(method)}"
        )


def _validate_run_alignment(runs: Sequence[RunMetrics]) -> None:
    canonical = runs[0]
    task_ids = set(canonical.metrics)
    for run in runs[1:]:
        if set(run.metrics) != task_ids:
            raise ValueError(f"run task IDs differ: {canonical.path} vs {run.path}")
        if run.graph_ids != canonical.graph_ids:
            raise ValueError(f"run graph IDs differ: {canonical.path} vs {run.path}")
        if run.memory_modes != canonical.memory_modes:
            raise ValueError(f"run memory modes differ: {canonical.path} vs {run.path}")
        if (
            run.test_artifact_digest != canonical.test_artifact_digest
            or run.test_payload_digests != canonical.test_payload_digests
        ):
            raise ValueError(
                f"run test artifacts differ: {canonical.path} vs {run.path}"
            )


def _run_test_artifact_identity(path: Path) -> tuple[str, dict[str, str]]:
    manifest = _mapping(_read_yaml(path / "assets" / "manifest.yaml"), "asset manifest")
    assets = _mapping_list(manifest.get("assets"), "asset manifest.assets")
    matches = [
        asset
        for asset in assets
        if asset.get("kind") == "dataset"
        and isinstance(asset.get("origin"), Mapping)
        and cast(Mapping[str, object], asset["origin"]).get("split") == "test"
    ]
    if len(matches) != 1:
        raise ValueError(f"run={path} must identify exactly one test dataset artifact")
    asset = matches[0]
    required_roles = {"labels", "tasks", "provenance_graphs"}
    payload_digests = {
        _required_string(
            payload.get("role"), "test asset.payload.role"
        ): _required_string(payload.get("digest"), "test asset.payload.digest")
        for payload in _mapping_list(asset.get("payloads"), "test asset.payloads")
        if payload.get("role") in required_roles
    }
    if set(payload_digests) != required_roles:
        raise ValueError(f"run={path} test asset lacks required payload identities")
    return _required_string(asset.get("digest"), "test asset.digest"), payload_digests


def _run_group_identity(runs: Mapping[int, RunMetrics]) -> dict[str, object]:
    first = runs[min(runs)]
    return {
        "method": first.method,
        "variant": first.variant,
        "seeds": sorted(runs),
        "runs": {str(seed): str(runs[seed].path) for seed in sorted(runs)},
    }


def _feature_counts(features: Iterable[TaskFeature]) -> dict[str, object]:
    values = list(features)
    return {
        "gold_span_count": _counts(str(value.gold_span_count) for value in values),
        "gold_event_count": _counts(str(value.gold_event_count) for value in values),
        "minimum_flat_chunks": _counts(
            str(value.minimum_flat_chunks) for value in values
        ),
        "fragmentation_bins": _counts(value.fragmentation_bin for value in values),
        "provenance_bins": _counts(
            value.provenance_bin for value in values if value.provenance_bin is not None
        ),
        "provenance_distance_eligible_tasks": sum(
            value.provenance_bin is not None for value in values
        ),
    }


def _write_summary_csv(path: Path, comparisons: Mapping[str, object]) -> None:
    rows: list[dict[str, object]] = []
    for comparison_name, raw_comparison in comparisons.items():
        comparison = _mapping(raw_comparison, f"comparisons.{comparison_name}")
        for raw_bin in cast(Sequence[object], comparison["bins"]):
            bin_result = _mapping(raw_bin, "comparison bin")
            metrics = _mapping(bin_result.get("metrics"), "comparison bin.metrics")
            for metric_name, raw_metric in metrics.items():
                metric = _mapping(raw_metric, f"metric.{metric_name}")
                ci = cast(Sequence[object], metric["ci_95"])
                rows.append(
                    {
                        "comparison": comparison_name,
                        "bin_name": comparison["bin_name"],
                        "bin": bin_result["bin"],
                        "task_count": bin_result["task_count"],
                        "trajectory_count": bin_result["trajectory_count"],
                        "metric": metric_name,
                        "baseline_mean": metric["baseline_mean"],
                        "method_mean": metric["method_mean"],
                        "mean_delta": metric["mean_delta"],
                        "ci_95_low": ci[0],
                        "ci_95_high": ci[1],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, comparisons: Mapping[str, object]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "matplotlib is required to render the mechanism figure"
        ) from error

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
    panels = (
        (
            axes[0],
            _mapping(comparisons["segmentation"], "segmentation"),
            "(a) Segmentation: PU Dense-FT − Flat",
            {"1": "1", "2": "2", "3+": "3+"},
            "Minimum flat chunks covering gold",
        ),
        (
            axes[1],
            _mapping(comparisons["graph_residual"], "graph_residual"),
            "(b) Graph residual: R-GCN − exact seed",
            {"1": "1", "2": "2", "3+": "3+", "disconnected": "Disc."},
            "Maximum observable provenance hops",
        ),
    )
    styles = {
        "Coverage@1024 Tokens": ("#0072B2", "o", "Coverage@1024"),
        "Full Support@2048 Tokens": ("#D55E00", "s", "Full Support@2048"),
    }
    for axis, comparison, title, labels, xlabel in panels:
        bins = [
            _mapping(value, "comparison bin")
            for value in cast(Sequence[object], comparison["bins"])
        ]
        x_values = list(range(len(bins)))
        for metric_name, (color, marker, legend_label) in styles.items():
            means: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for bin_result in bins:
                metric = _mapping(
                    _mapping(bin_result["metrics"], "bin.metrics")[metric_name],
                    f"bin.metrics.{metric_name}",
                )
                mean = 100.0 * _required_number(metric["mean_delta"], "mean_delta")
                ci = cast(Sequence[object], metric["ci_95"])
                means.append(mean)
                lower.append(mean - 100.0 * _required_number(ci[0], "ci_95[0]"))
                upper.append(100.0 * _required_number(ci[1], "ci_95[1]") - mean)
            axis.errorbar(
                x_values,
                means,
                yerr=[lower, upper],
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.35,
                capsize=2.5,
                label=legend_label,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xticks(
            x_values,
            [
                f"{labels[str(bin_result['bin'])]}\n$n$={cast(int, bin_result['task_count'])}"
                for bin_result in bins
            ],
        )
        axis.set_title(title, fontsize=9.2)
        axis.set_xlabel(xlabel, fontsize=8.5)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.tick_params(labelsize=7.8)
    axes[0].set_ylabel("Mean paired gain (percentage points)", fontsize=8.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8.2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _write_dispersion_figure(
    path: Path, comparison: Mapping[str, object]
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "matplotlib is required to render the dispersion figure"
        ) from error

    figure, axis = plt.subplots(1, 1, figsize=(4.0, 2.75), constrained_layout=True)
    styles = {
        "Coverage@1024 Tokens": ("#0072B2", "o", "Coverage@1024"),
        "Full Support@2048 Tokens": ("#D55E00", "s", "Full Support@2048"),
    }
    bins = [
        _mapping(value, "comparison bin")
        for value in cast(Sequence[object], comparison["bins"])
    ]
    x_values = list(range(len(bins)))
    for metric_name, (color, marker, legend_label) in styles.items():
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for bin_result in bins:
            metric = _mapping(
                _mapping(bin_result["metrics"], "bin.metrics")[metric_name],
                f"bin.metrics.{metric_name}",
            )
            mean = 100.0 * _required_number(metric["mean_delta"], "mean_delta")
            ci = cast(Sequence[object], metric["ci_95"])
            means.append(mean)
            lower.append(mean - 100.0 * _required_number(ci[0], "ci_95[0]"))
            upper.append(100.0 * _required_number(ci[1], "ci_95[1]") - mean)
        axis.errorbar(
            x_values,
            means,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.35,
            capsize=2.5,
            label=legend_label,
        )
    axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xticks(
        x_values,
        [
            f"{bin_result['bin']}\n$n$={cast(int, bin_result['task_count'])}"
            for bin_result in bins
        ],
    )
    axis.set_xlabel("Distinct gold-bearing execution events", fontsize=8.5)
    axis.set_ylabel("Mean paired gain (percentage points)", fontsize=8.5)
    axis.set_title("Graph residual by evidence dispersion", fontsize=9.2)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.tick_params(labelsize=7.8)
    axis.legend(frameon=False, fontsize=8.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def _gold_atoms(
    gold: Sequence[SourceInterval],
    candidates: Sequence[Sequence[SourceInterval]],
) -> list[SourceInterval]:
    by_source: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for interval in gold:
        by_source.setdefault((interval.event_id, interval.json_pointer), []).append(
            (interval.start, interval.end)
        )
    atoms: list[SourceInterval] = []
    for (event_id, pointer), raw_gold_intervals in sorted(by_source.items()):
        merged_gold = _merge_intervals(raw_gold_intervals)
        boundaries = {value for interval in merged_gold for value in interval}
        for candidate in candidates:
            for interval in candidate:
                if (interval.event_id, interval.json_pointer) != (event_id, pointer):
                    continue
                for gold_start, gold_end in merged_gold:
                    start = max(interval.start, gold_start)
                    end = min(interval.end, gold_end)
                    if end > start:
                        boundaries.update((start, end))
        ordered = sorted(boundaries)
        for start, end in zip(ordered, ordered[1:]):
            if end <= start:
                continue
            if any(
                gold_start <= start and end <= gold_end
                for gold_start, gold_end in merged_gold
            ):
                atoms.append(SourceInterval(event_id, pointer, start, end))
    return atoms


def _coverage_mask(
    candidate: Sequence[SourceInterval], atoms: Sequence[SourceInterval]
) -> int:
    mask = 0
    for index, atom in enumerate(atoms):
        if any(
            interval.event_id == atom.event_id
            and interval.json_pointer == atom.json_pointer
            and interval.start <= atom.start
            and interval.end >= atom.end
            for interval in candidate
        ):
            mask |= 1 << index
    return mask


def _reachable_unions(masks: set[int]) -> set[int]:
    reachable = {0}
    for mask in masks:
        reachable.update(state | mask for state in tuple(reachable))
    return reachable


def _source_intervals(values: Sequence[Mapping[str, object]]) -> list[SourceInterval]:
    intervals: list[SourceInterval] = []
    for value in values:
        start = value.get("char_start")
        end = value.get("char_end")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError("source span char_start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool) or end <= start:
            raise ValueError("source span char_end must exceed char_start")
        intervals.append(
            SourceInterval(
                event_id=_required_string(
                    value.get("event_id"), "source span.event_id"
                ),
                json_pointer=_required_string(
                    value.get("json_pointer"), "source span.json_pointer"
                ),
                start=start,
                end=end,
            )
        )
    return intervals


def _merge_intervals(values: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(values):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _shortest_distances(
    adjacency: Mapping[str, set[str]], source: str
) -> dict[str, int]:
    distances = {source: 0}
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            pending.append(neighbor)
    return distances


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _mapping_list(value: object, path: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return [_mapping(item, path) for item in value]


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _required_number(value: object, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
