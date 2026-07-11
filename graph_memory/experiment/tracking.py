from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

from graph_memory.experiment.config import ResolvedExperimentConfig
from graph_memory.experiment.planning import StageInvocation
from graph_memory.experiment.state import StageRunSummary

ALLOWLISTED_ARTIFACT_ROLES = frozenset(
    {
        "resolved_config",
        "overrides",
        "stage_config",
        "stage_summary",
        "selected_config",
        "candidate_table",
        "main_table",
        "path_table",
        "efficiency_table",
        "ablation_table",
        "failure_cases",
        "report_image",
    }
)
MAX_TRACKED_ARTIFACT_BYTES = 10 * 1024 * 1024


class TrackingAdapter:
    def __init__(self, config: ResolvedExperimentConfig) -> None:
        self.config = config
        config.tracking.database.parent.mkdir(parents=True, exist_ok=True)
        config.tracking.artifact_root.mkdir(parents=True, exist_ok=True)
        self.client = MlflowClient(tracking_uri=config.tracking.tracking_uri)
        experiment = self.client.get_experiment_by_name(config.tracking.experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(
                config.tracking.experiment_name,
                artifact_location=config.tracking.artifact_root.as_uri(),
            )
            experiment = self.client.get_experiment(experiment_id)
        if Path(
            experiment.artifact_location.removeprefix("file://")
        ).as_posix().lower() != (config.tracking.artifact_root.as_posix().lower()):
            raise ValueError(
                "MLflow experiment artifact root mismatch: "
                f"expected={config.tracking.artifact_root} actual={experiment.artifact_location}"
            )
        self.experiment_id = experiment.experiment_id

    def start_parent(self, *, existing_run_id: str | None = None) -> str:
        if existing_run_id is not None:
            run = self.client.get_run(existing_run_id)
            if run.info.experiment_id != self.experiment_id:
                raise ValueError(
                    f"persisted MLflow parent belongs to experiment={run.info.experiment_id}, "
                    f"expected={self.experiment_id}"
                )
            self.client.update_run(existing_run_id, status="RUNNING")
            run_id = existing_run_id
        else:
            run = self.client.create_run(
                self.experiment_id,
                tags={
                    MLFLOW_RUN_NAME: self.config.name,
                    "graph_memory.run_kind": "parent",
                },
            )
            run_id = run.info.run_id
        self.log_parent_metadata(run_id)
        return run_id

    def start_child(self, parent_run_id: str, invocation: StageInvocation) -> str:
        tags = {
            MLFLOW_PARENT_RUN_ID: parent_run_id,
            MLFLOW_RUN_NAME: invocation.identifier,
            "graph_memory.run_kind": "stage",
            "graph_memory.stage": invocation.stage,
            "graph_memory.identifier": invocation.identifier,
        }
        if invocation.method is not None:
            tags["graph_memory.method"] = invocation.method.value
        if invocation.split is not None:
            tags["graph_memory.split"] = invocation.split
        if invocation.variant is not None:
            tags["graph_memory.variant"] = invocation.variant
        return self.client.create_run(self.experiment_id, tags=tags).info.run_id

    def finish(self, run_id: str, *, status: str) -> None:
        self.client.set_terminated(run_id, status=status)

    def log_parent_metadata(self, run_id: str) -> None:
        for key, value in _flatten("config", self.config.normalized()).items():
            self.client.log_param(run_id, key, value)
        for key, value in _provenance().items():
            self.client.set_tag(run_id, key, value)

    def log_stage(
        self,
        run_id: str,
        invocation: StageInvocation,
        summary: StageRunSummary,
        *,
        resolved_config: Path,
        overrides: Path,
    ) -> None:
        for key, value in _flatten(
            "stage_config",
            invocation.config.model_dump(mode="json", by_alias=True),
        ).items():
            self.client.log_param(run_id, key, value)
        self.client.set_tag(run_id, "graph_memory.status", summary.status)
        if summary.error is not None:
            self.client.set_tag(run_id, "graph_memory.error_type", summary.error.type)
            self.client.set_tag(
                run_id, "graph_memory.error_message", summary.error.message
            )
        for name, value in summary.timings.items():
            self.client.log_metric(run_id, _metric_name(f"timing.{name}"), float(value))
        for name, value in summary.counts.items():
            if _is_metric(value):
                self.client.log_metric(
                    run_id, _metric_name(f"count.{name}"), float(value)
                )

        summary_path = _summary_path(invocation)
        curated = (
            ("resolved_config", resolved_config),
            ("overrides", overrides),
            ("stage_config", invocation.config_path),
            ("stage_summary", summary_path),
        )
        for role, path in curated:
            self._log_artifact_or_metadata(run_id, role, path, kind="file")
        for artifact in invocation.outputs:
            self._log_artifact_or_metadata(
                run_id,
                artifact.role,
                artifact.path,
                kind=artifact.kind,
            )
        self._log_local_metrics(run_id, invocation)

    def _log_artifact_or_metadata(
        self,
        run_id: str,
        role: str,
        path: Path,
        *,
        kind: str,
    ) -> None:
        size = _path_size(path)
        prefix = f"artifact.{_metric_name(role)}"
        self.client.log_param(run_id, f"{prefix}.path", str(path.resolve()))
        self.client.log_param(run_id, f"{prefix}.kind", kind)
        self.client.log_param(run_id, f"{prefix}.size_bytes", str(size))
        if role not in ALLOWLISTED_ARTIFACT_ROLES:
            self.client.set_tag(run_id, f"{prefix}.upload", "prohibited")
            return
        if kind != "file":
            raise ValueError(f"allowlisted MLflow artifact role={role} must be a file")
        if not path.is_file():
            raise FileNotFoundError(f"allowlisted MLflow artifact missing: {path}")
        if size > MAX_TRACKED_ARTIFACT_BYTES:
            raise ValueError(
                f"allowlisted MLflow artifact exceeds size limit role={role} size={size}: {path}"
            )
        self.client.log_artifact(run_id, str(path), artifact_path=role)

    def _log_local_metrics(self, run_id: str, invocation: StageInvocation) -> None:
        if invocation.stage == "train":
            metrics = next(
                (
                    output.path
                    for output in invocation.outputs
                    if output.role == "train_metrics"
                ),
                None,
            )
            if metrics is not None and metrics.is_file():
                for index, line in enumerate(
                    metrics.read_text(encoding="utf-8").splitlines()
                ):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    step_value = record.get("epoch", index)
                    step = int(step_value) if isinstance(step_value, int) else index
                    for name, value in record.items():
                        if name != "epoch" and _is_metric(value):
                            self.client.log_metric(
                                run_id,
                                _metric_name(f"train.{name}"),
                                float(value),
                                step=step,
                            )
        if invocation.stage in {"evaluate", "aggregate"}:
            for output in invocation.outputs:
                if output.path.suffix.lower() != ".csv" or not output.path.is_file():
                    continue
                with output.path.open("r", encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != 1:
                    continue
                for name, value in rows[0].items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(number):
                        self.client.log_metric(
                            run_id,
                            _metric_name(f"{invocation.stage}.{name}"),
                            number,
                        )
        if invocation.stage == "tune":
            self._log_tuning_selection(run_id, invocation)

    def _log_tuning_selection(self, run_id: str, invocation: StageInvocation) -> None:
        selected_path = next(
            (
                output.path
                for output in invocation.outputs
                if output.role == "selected_config"
            ),
            None,
        )
        candidates_path = next(
            (
                output.path
                for output in invocation.outputs
                if output.role == "candidate_table"
            ),
            None,
        )
        if selected_path is None or not selected_path.is_file():
            return
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if isinstance(selected, dict):
            for key, value in _flatten("tune.selected", selected).items():
                self.client.log_param(run_id, key, value)
        if candidates_path is None or not candidates_path.is_file():
            return
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        if not isinstance(candidates, list):
            return
        selected_row = next(
            (
                row
                for row in candidates
                if isinstance(row, dict) and row.get("config") == selected
            ),
            None,
        )
        if selected_row is None:
            return
        for name, value in selected_row.items():
            if name != "config" and _is_metric(value):
                self.client.log_metric(
                    run_id,
                    _metric_name(f"tune.best.{name}"),
                    float(value),
                )


def _summary_path(invocation: StageInvocation) -> Path:
    from graph_memory.experiment.state import summary_path_for

    return summary_path_for(invocation)


def _flatten(prefix: str, value: object) -> dict[str, str]:
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, item in value.items():
            result.update(_flatten(f"{prefix}.{key}", item))
        return result
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, separators=(",", ":"))}
    if value is None:
        return {prefix: "null"}
    return {prefix: str(value)}


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _is_metric(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _metric_name(value: str) -> str:
    return value.replace(" ", "_").replace("@", "_at_")


def _provenance() -> dict[str, str]:
    revision = _git_output("rev-parse", "HEAD")
    dirty = bool(_git_output("status", "--porcelain"))
    versions = {
        package: _package_version(package)
        for package in ("hydra-core", "pydantic", "mlflow", "torch")
    }
    return {
        "graph_memory.git_revision": revision or "unknown",
        "graph_memory.git_dirty": str(dirty).lower(),
        "graph_memory.python": platform.python_version(),
        "graph_memory.platform": platform.platform(),
        **{f"graph_memory.version.{key}": value for key, value in versions.items()},
    }


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


__all__ = [
    "ALLOWLISTED_ARTIFACT_ROLES",
    "MAX_TRACKED_ARTIFACT_BYTES",
    "TrackingAdapter",
]
