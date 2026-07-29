from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_memory.analysis import analyze_main_results
from graph_memory.io import write_json

# Metrics that are defined per query and averaged into the aggregate row. These
# are the columns emitted per task by the evaluation suite; aggregate-only
# metrics (micro edge precision/recall, etc.) are intentionally excluded.
_PER_TASK_METRICS = (
    "Recall@2",
    "Recall@5",
    "Recall@10",
    "Evidence F1@5",
    "Evidence F1@10",
    "Full Support@5",
    "Full Support@10",
    "MRR",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate main-results runs into per-method mean/std (trainable) "
            "or single value (deterministic) plus paired bootstrap 95% CIs "
            "against a baseline. All runs must share the same fixed test split."
        )
    )
    parser.add_argument(
        "--run",
        dest="runs",
        type=Path,
        action="append",
        required=True,
        help="Run output directory (repeatable). Each must contain "
        "workflow/summary.yaml and metrics/per_task.jsonl.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline method name to compare every other method against.",
    )
    parser.add_argument(
        "--trainable",
        action="append",
        default=[],
        help="Method name that is trainable (runs multiple seeds). Repeatable. "
        "Methods not listed are treated as deterministic (single run).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args()

    trainable_methods = set(args.trainable)
    rows = [_row_from_run(run, trainable_methods=trainable_methods) for run in args.runs]
    result = analyze_main_results(
        rows,
        baseline_method=args.baseline,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(args.output, result)
    return 0


def _row_from_run(run: Path, *, trainable_methods: set[str]) -> dict[str, object]:
    summary_path = run / "workflow" / "summary.yaml"
    per_task_path = run / "metrics" / "per_task.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    if not per_task_path.is_file():
        raise FileNotFoundError(f"missing per-task metrics: {per_task_path}")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = yaml.safe_load(stream)
    method = summary["method"]
    seed = summary["seed"]

    per_task: dict[str, dict[str, float]] = {}
    with per_task_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            task_id = record["task_id"]
            per_task[task_id] = {
                metric: float(record[metric])
                for metric in _PER_TASK_METRICS
                if metric in record
            }
    if not per_task:
        raise ValueError(f"no per-task rows in {per_task_path}")

    metrics = _aggregate_from_per_task(per_task)
    return {
        "method": method,
        "trainable": method in trainable_methods,
        "seed": seed,
        "metrics": metrics,
        "per_task": per_task,
    }


def _aggregate_from_per_task(
    per_task: dict[str, dict[str, float]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    task_count = len(per_task)
    for metric in _PER_TASK_METRICS:
        values = [
            task_metrics[metric]
            for task_metrics in per_task.values()
            if metric in task_metrics
        ]
        if len(values) == task_count and task_count > 0:
            metrics[metric] = sum(values) / task_count
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
