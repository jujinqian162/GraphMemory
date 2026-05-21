from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from graph_memory.io import write_json
from graph_memory.types import (
    GraphStatistics,
    JsonObject,
    MemoryGraph,
    RankedNodeDebugRecord,
    RankedNodeRecord,
    RunSummary,
    ScoreBreakdown,
    ScoreDebugRecord,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def collect_environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def build_run_summary(
    *,
    script: str,
    started_at: str,
    finished_at: str,
    status: str,
    effective_config: JsonObject,
    inputs: JsonObject,
    outputs: JsonObject,
    counts: JsonObject,
    timings: JsonObject,
    environment: dict[str, str] | None = None,
    notes: list[str] | None = None,
    error: str | None = None,
) -> RunSummary:
    summary: RunSummary = {
        "script": script,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "effective_config": effective_config,
        "inputs": inputs,
        "outputs": outputs,
        "counts": counts,
        "timings": timings,
        "environment": environment or collect_environment(),
        "notes": notes or [],
    }
    if error is not None:
        summary["error"] = error
    return summary


def write_run_summary(path: str | Path, summary: RunSummary) -> None:
    write_json(path, summary)


def graph_statistics(
    graphs: list[MemoryGraph], *, split: str | None = None, graph_config: JsonObject | None = None
) -> GraphStatistics:
    edge_counts: Counter[str] = Counter()
    total_nodes = 0
    total_edges = 0
    isolated_memory_nodes = 0

    for graph in graphs:
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        total_nodes += len(nodes)
        total_edges += len(edges)
        for edge in edges:
            edge_counts[str(edge.get("edge_type"))] += 1

        incident_node_ids: set[str] = set()
        for edge in edges:
            incident_node_ids.add(str(edge.get("source")))
            incident_node_ids.add(str(edge.get("target")))
        for node in nodes:
            if node.get("id") != "q" and node.get("id") not in incident_node_ids:
                isolated_memory_nodes += 1

    num_graphs = len(graphs)
    stats: GraphStatistics = {
        "num_graphs": num_graphs,
        "avg_nodes": total_nodes / num_graphs if num_graphs else 0.0,
        "avg_edges": total_edges / num_graphs if num_graphs else 0.0,
        "edge_counts_by_type": dict(sorted(edge_counts.items())),
        "isolated_memory_nodes": isolated_memory_nodes,
    }
    if split is not None:
        stats["split"] = split
    if graph_config is not None:
        stats["graph_config"] = graph_config
    return stats


def config_digest(config: JsonObject) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_score_debug_record(
    *,
    task_id: str,
    method: str,
    top_k: int,
    ranked_nodes: list[RankedNodeRecord],
    score_breakdown: ScoreBreakdown,
    split: str | None = None,
    config: JsonObject | None = None,
) -> ScoreDebugRecord:
    debug_ranked_nodes: list[RankedNodeDebugRecord] = []
    for ranked_node in ranked_nodes[:top_k]:
        debug_node: RankedNodeDebugRecord = {"node_id": ranked_node["node_id"], "score": ranked_node["score"]}
        node_id = ranked_node["node_id"]
        if node_id in score_breakdown:
            debug_node["score_components"] = score_breakdown[node_id]
        debug_ranked_nodes.append(debug_node)

    record: ScoreDebugRecord = {
        "debug_type": "score_breakdown",
        "task_id": task_id,
        "method": method,
        "top_k": top_k,
        "ranked_nodes": debug_ranked_nodes,
    }
    if split is not None:
        record["split"] = split
    if config is not None:
        record["config_digest"] = config_digest(config)
    return record
