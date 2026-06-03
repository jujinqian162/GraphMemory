from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from graph_memory.contracts.common import JsonObject
from graph_memory.contracts.observability import RunSummary
from graph_memory.infrastructure.io import write_json
from graph_memory.infrastructure.runtime_environment import collect_environment


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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


__all__ = ["build_run_summary", "now_iso", "write_run_summary"]
