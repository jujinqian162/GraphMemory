from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TypeGuard

import mlflow

from graph_memory.experiment.config import ResolvedExperimentConfig
from graph_memory.experiment.results import FinalExperimentResult


FINAL_METRIC_KEYS = {
    "Recall@2": "final.recall_at_2",
    "Recall@5": "final.recall_at_5",
    "Recall@10": "final.recall_at_10",
    "Evidence F1@5": "final.evidence_f1_at_5",
    "Evidence F1@10": "final.evidence_f1_at_10",
    "Full Support@5": "final.full_support_at_5",
    "Full Support@10": "final.full_support_at_10",
    "MRR": "final.mrr",
    "Connected Evidence Recall@5": "final.connected_evidence_recall_at_5",
    "Connected Evidence Recall@10": "final.connected_evidence_recall_at_10",
    "Query-Evidence Connectivity@10": "final.query_evidence_connectivity_at_10",
    "Path Recall@10": "final.path_recall_at_10",
    "Edge Recall@10": "final.edge_recall_at_10",
    "Edge Precision@10": "final.edge_precision_at_10",
    "Edge F1@10": "final.edge_f1_at_10",
    "Abstention Rate": "final.abstention_rate",
    "Avg Retrieved Nodes": "final.avg_retrieved_nodes",
    "Avg Retrieved Edges": "final.avg_retrieved_edges",
}
_KNOWN_EFFICIENCY_COLUMNS = {
    "Retrieval Latency / Query",
    "Index Build Time",
    "Graph Construction Time",
    "Memory Size",
}


def configure_tracking(config: ResolvedExperimentConfig) -> None:
    config.tracking.database.parent.mkdir(parents=True, exist_ok=True)
    config.tracking.artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.tracking.tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.tracking.experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            config.tracking.experiment_name,
            artifact_location=config.tracking.artifact_root.as_uri(),
        )
    mlflow.set_experiment(config.tracking.experiment_name)


def log_experiment_result(
    config: ResolvedExperimentConfig,
    result: FinalExperimentResult,
    *,
    run_output: Path,
    prefect_flow_run_id: str,
) -> None:
    if mlflow.active_run() is None:
        raise RuntimeError("experiment result logging requires one active MLflow run")
    mlflow.log_params(
        {
            key: _param_value(value)
            for key, value in _flatten("", config.normalized()).items()
        }
    )
    mlflow.set_tags(
        {
            "graph_memory.method": result.method,
            "graph_memory.variant": result.variant or "none",
            "graph_memory.dataset": config.dataset.name,
            "graph_memory.profile": config.profile,
            "graph_memory.seed": str(config.seed),
            "graph_memory.study": config.name,
            "graph_memory.prefect_flow_run_id": prefect_flow_run_id,
        }
    )
    metric_row = result.evaluation.metric_rows[0].model_dump(
        mode="json", by_alias=True
    )
    metrics: dict[str, float] = {}
    for column, value in metric_row.items():
        if column == "Method":
            continue
        metric_key = FINAL_METRIC_KEYS.get(column)
        number = _finite_number(value)
        if metric_key is not None and number is not None:
            metrics[metric_key] = number
        elif column not in _KNOWN_EFFICIENCY_COLUMNS and number is not None:
            raise ValueError(f"unknown numeric final metric column={column!r}")
    if result.benchmark is not None:
        metrics.update(result.benchmark.metrics)
    if metrics:
        mlflow.log_metrics(metrics)

    mlflow.log_dict(
        {"assets": [asset.model_dump(mode="json") for asset in result.assets]},
        "assets/manifest.json",
    )
    if result.model is not None:
        mlflow.set_tags(
            {
                "graph_memory.training_asset_origin": result.model.artifact.digest,
            }
        )
        for index, record in enumerate(result.model.training_history):
            step_value = record.get("epoch")
            step = step_value if isinstance(step_value, int) else index
            epoch_metrics = {
                f"train.{_metric_component(key)}": float(value)
                for key, value in record.items()
                if key != "epoch" and _is_metric(value)
            }
            if epoch_metrics:
                mlflow.log_metrics(epoch_metrics, step=step)
    mlflow.log_artifacts(str(run_output))


def _flatten(prefix: str, value: object) -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            nested = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(nested, item))
        return result
    return {prefix: value}


def _param_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _is_metric(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _metric_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


__all__ = [
    "FINAL_METRIC_KEYS",
    "configure_tracking",
    "log_experiment_result",
]
