from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

from graph_memory.experiment.config import ResolvedExperimentConfig
from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.layout import MultirunIdentity, RunLayout
from graph_memory.experiment.state import FailedStageRunSummary, StageRunSummary
from graph_memory.registry.retrieval import RetrievalMethodId

MAX_TRACKED_ARTIFACT_BYTES = 10 * 1024 * 1024
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
    "Retrieval Latency / Query": "final.retrieval_latency_per_query",
    "Index Build Time": "final.index_build_time",
    "Graph Construction Time": "final.graph_construction_time",
    "Memory Size": "final.memory_size",
    "Avg Retrieved Nodes": "final.avg_retrieved_nodes",
    "Avg Retrieved Edges": "final.avg_retrieved_edges",
}
_IDENTITY_COLUMNS = frozenset({"Method", "Variant"})
_CURATED_OUTPUT_PATHS = {
    "selected_config": "tuning",
    "candidate_table": "tuning",
    "train_pair_summary": "training",
    "train_metrics": "training",
    "metrics": "evaluation",
    "failure_cases": "evaluation",
}
_METHOD_CONFIG_FIELDS = {
    "encoder": "encoder",
    "hard_dense_encoder": "encoder",
    "pairs": "pairs",
    "sampling": "pairs",
    "train": "train",
    "scoring": "scoring",
    "search_space": "tuning.search_space",
}


@dataclass(frozen=True)
class BaselineIdentity:
    method: RetrievalMethodId
    variant: str | None = None

    @property
    def tag_variant(self) -> str:
        return self.variant or "ordinary"

    @property
    def run_name(self) -> str:
        if self.variant is None:
            return self.method.value
        return f"{self.method.value}/{self.variant}"


class TrackingAdapter:
    def __init__(self, config: ResolvedExperimentConfig, layout: RunLayout) -> None:
        self.config = config
        self.layout = layout
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
        tracked_artifact_root = Path(
            url2pathname(urlparse(experiment.artifact_location).path)
        ).resolve()
        if tracked_artifact_root != config.tracking.artifact_root.resolve():
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
                    MLFLOW_RUN_NAME: self._parent_run_name(),
                    "graph_memory.run_kind": "parent",
                },
            )
            run_id = run.info.run_id
        self._initialize_parent(run_id)
        return run_id

    def get_or_create_baseline_child(
        self,
        parent_run_id: str,
        identity: BaselineIdentity,
    ) -> str:
        matches = [
            run
            for run in self.client.search_runs(
                [self.experiment_id],
                filter_string=f"tags.`{MLFLOW_PARENT_RUN_ID}` = '{parent_run_id}'",
            )
            if run.data.tags.get("graph_memory.run_kind") == "baseline"
            and run.data.tags.get("graph_memory.method") == identity.method.value
            and run.data.tags.get("graph_memory.variant") == identity.tag_variant
        ]
        if len(matches) > 1:
            raise ValueError(
                "duplicate MLflow baseline children for "
                f"parent={parent_run_id} baseline={identity.run_name}"
            )
        if matches:
            run_id = matches[0].info.run_id
            self.client.update_run(run_id, status="RUNNING")
            return run_id
        run = self.client.create_run(
            self.experiment_id,
            tags={
                MLFLOW_PARENT_RUN_ID: parent_run_id,
                MLFLOW_RUN_NAME: identity.run_name,
                "graph_memory.run_kind": "baseline",
                "graph_memory.method": identity.method.value,
                "graph_memory.variant": identity.tag_variant,
            },
        )
        run_id = run.info.run_id
        self._log_stable_params(
            run_id,
            {
                "baseline.method": identity.method.value,
                "baseline.variant": identity.tag_variant,
            },
        )
        return run_id

    def log_parent_stage(
        self,
        run_id: str,
        invocation: StageInvocation,
        summary: StageRunSummary,
    ) -> None:
        if invocation.method is not None:
            raise ValueError(f"method-owned stage cannot be logged on parent: {invocation.identifier}")
        self._log_summary_tags(run_id, invocation, summary)
        self._log_curated_file(
            run_id,
            invocation.summary_path,
            artifact_path=f"workflow/{invocation.stage}",
        )
        for artifact in (*invocation.inputs, *invocation.outputs):
            self._log_artifact_metadata(run_id, invocation, artifact.role, artifact.path, artifact.kind)
        if invocation.stage == "aggregate":
            for output in invocation.outputs:
                if output.role.endswith("_table") and output.path.is_file():
                    self._log_curated_file(run_id, output.path, artifact_path="results")

    def log_baseline_stage(
        self,
        run_id: str,
        identity: BaselineIdentity,
        invocation: StageInvocation,
        summary: StageRunSummary,
        *,
        dependency: bool = False,
    ) -> None:
        logged_tag = f"graph_memory.logged.{_tag_component(invocation.identifier)}"
        if self.client.get_run(run_id).data.tags.get(logged_tag) == "true":
            return
        self._log_summary_tags(run_id, invocation, summary)
        namespace = (
            f"dependencies/{invocation.method.value}"
            if dependency and invocation.method is not None
            else "workflow"
        )
        self._log_curated_file(
            run_id,
            invocation.summary_path,
            artifact_path=f"{namespace}/{invocation.stage}",
        )
        for artifact in (*invocation.inputs, *invocation.outputs):
            self._log_artifact_metadata(run_id, invocation, artifact.role, artifact.path, artifact.kind)
        if not dependency:
            self._log_effective_method_parameters(run_id, invocation)
            self._log_curated_stage_outputs(run_id, invocation)
            if invocation.stage == "train":
                self._log_training_metrics(run_id, invocation)
            elif invocation.stage == "evaluate":
                self._log_final_metrics(run_id, identity)
            elif invocation.stage == "tune":
                self._log_tuning_selection(run_id, invocation)
        if summary.status == "success":
            self.client.set_tag(run_id, logged_tag, "true")

    def finalize_baseline(self, run_id: str, identity: BaselineIdentity) -> None:
        method_config = self.config.method_configs.get(identity.method)
        effective = {
            key: value
            for key, value in self.client.get_run(run_id).data.params.items()
            if not key.startswith("baseline.")
        }
        self.client.log_dict(
            run_id,
            {
                "method": identity.method.value,
                "variant": identity.variant,
                "configured": method_config.model_dump(mode="json", by_alias=True),
                "effective_parameters": effective,
            },
            "config/method.yaml",
        )
        self._log_final_metrics(run_id, identity)

    def reference_dependency(
        self,
        run_id: str,
        *,
        dependency: BaselineIdentity,
        dependency_run_id: str,
    ) -> None:
        prefix = f"dependency.{dependency.method.value}"
        self.client.set_tag(run_id, f"{prefix}.baseline", dependency.run_name)
        self.client.set_tag(run_id, f"{prefix}.run_id", dependency_run_id)

    def finalize_parent(self, run_id: str) -> None:
        rows = _merged_result_rows(self.layout)
        if rows:
            self.client.set_tag(run_id, "mlflow.note.content", _render_result_table(rows))
        else:
            self.client.set_tag(
                run_id,
                "mlflow.note.content",
                f"Experiment {self.config.name}: no aggregate result table is available for this stage-bounded run.",
            )
        for kind in ("main", "path", "efficiency", "ablation"):
            path = self.layout.table(kind)
            if path.is_file():
                self._log_curated_file(run_id, path, artifact_path="results")

    def finish(self, run_id: str, *, status: str) -> None:
        self.client.set_terminated(run_id, status=status)

    def _initialize_parent(self, run_id: str) -> None:
        if self.client.get_run(run_id).data.tags.get("graph_memory.parent.initialized") == "true":
            return
        params: dict[str, object] = {
            "name": self.config.name,
            "dataset": self.config.dataset.name,
            "profile": self.config.profile,
            "seed": self.config.seed,
            "device": self.config.device,
            "methods": [method.value for method in self.config.methods],
            "top_k": self.config.top_k,
            "stages.from": self.config.stages.from_stage,
            "stages.to": self.config.stages.to_stage,
            "cache.enabled": self.config.cache.enabled,
            "ablation.variants": self.config.ablation.variants,
            "ablation.only": self.config.ablation.only,
        }
        if isinstance(self.layout.identity, MultirunIdentity):
            params.update(
                {
                    "multirun.job_num": self.layout.identity.job_num,
                    "multirun.suffix": self.layout.identity.suffix,
                }
            )
        self._log_stable_params(run_id, params)
        for key, value in _provenance().items():
            self.client.set_tag(run_id, key, value)
        self._log_curated_file(run_id, self.layout.resolved_config, artifact_path="config")
        self._log_curated_file(run_id, self.layout.overrides, artifact_path="config")
        self.client.set_tag(run_id, "graph_memory.parent.initialized", "true")

    def _parent_run_name(self) -> str:
        if isinstance(self.layout.identity, MultirunIdentity):
            return f"{self.config.name}/{self.layout.run_dir.name}"
        return self.config.name

    def _log_summary_tags(
        self,
        run_id: str,
        invocation: StageInvocation,
        summary: StageRunSummary,
    ) -> None:
        prefix = f"graph_memory.stage.{_tag_component(invocation.identifier)}"
        self.client.set_tag(run_id, f"{prefix}.status", summary.status)
        self.client.set_tag(
            run_id,
            f"{prefix}.counts",
            json.dumps(summary.counts, sort_keys=True, separators=(",", ":")),
        )
        self.client.set_tag(
            run_id,
            f"{prefix}.timings",
            json.dumps(summary.timings, sort_keys=True, separators=(",", ":")),
        )
        if isinstance(summary, FailedStageRunSummary):
            self.client.set_tag(run_id, f"{prefix}.error_type", summary.error.type)
            self.client.set_tag(run_id, f"{prefix}.error_message", summary.error.message)

    def _log_artifact_metadata(
        self,
        run_id: str,
        invocation: StageInvocation,
        role: str,
        path: Path,
        kind: str,
    ) -> None:
        prefix = (
            f"artifact.{_tag_component(invocation.identifier)}.{_tag_component(role)}"
        )
        self.client.set_tag(run_id, f"{prefix}.path", str(path.resolve()))
        self.client.set_tag(run_id, f"{prefix}.kind", kind)
        self.client.set_tag(run_id, f"{prefix}.size_bytes", str(_path_size(path)))
        if role not in _CURATED_OUTPUT_PATHS:
            self.client.set_tag(run_id, f"{prefix}.upload", "prohibited")

    def _log_curated_stage_outputs(
        self,
        run_id: str,
        invocation: StageInvocation,
    ) -> None:
        for output in invocation.outputs:
            artifact_path = _CURATED_OUTPUT_PATHS.get(output.role)
            if artifact_path is not None and output.path.is_file():
                self._log_curated_file(run_id, output.path, artifact_path=artifact_path)

    def _log_curated_file(
        self,
        run_id: str,
        path: Path,
        *,
        artifact_path: str,
    ) -> None:
        if not path.is_file():
            return
        size = path.stat().st_size
        if size > MAX_TRACKED_ARTIFACT_BYTES:
            raise ValueError(
                f"curated MLflow artifact exceeds size limit size={size}: {path}"
            )
        self.client.log_artifact(run_id, str(path), artifact_path=artifact_path)

    def _log_effective_method_parameters(
        self,
        run_id: str,
        invocation: StageInvocation,
    ) -> None:
        raw = invocation.config.model_dump(mode="json", by_alias=True)
        projected: dict[str, object] = {}
        for source, target in _METHOD_CONFIG_FIELDS.items():
            if source in raw:
                projected[target] = raw[source]
        self._log_stable_params(run_id, _flatten("", projected))

    def _log_tuning_selection(self, run_id: str, invocation: StageInvocation) -> None:
        selected_path = _output_path(invocation, "selected_config")
        candidates_path = _output_path(invocation, "candidate_table")
        if selected_path is None or not selected_path.is_file():
            return
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if isinstance(selected, dict):
            self._log_stable_params(run_id, _flatten("tuning.selected", selected))
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
        if selected_row is not None:
            for name, value in selected_row.items():
                if name != "config":
                    self.client.set_tag(
                        run_id,
                        f"tuning.best.{_tag_component(name)}",
                        _param_value(value),
                    )

    def _log_training_metrics(self, run_id: str, invocation: StageInvocation) -> None:
        path = _output_path(invocation, "train_metrics")
        if path is None or not path.is_file():
            return
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            epoch = record.get("epoch")
            step = epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else index
            for name, value in record.items():
                if name != "epoch" and _is_metric(value):
                    self.client.log_metric(
                        run_id,
                        f"train.{_metric_component(name)}",
                        float(value),
                        step=step,
                    )

    def _log_final_metrics(self, run_id: str, identity: BaselineIdentity) -> None:
        if self.client.get_run(run_id).data.tags.get("graph_memory.final.logged") == "true":
            return
        path = self.layout.metric(identity.method, variant=identity.variant)
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError(
                f"baseline metric artifact must contain exactly one row: {path}"
            )
        row = rows[0]
        for column, value in row.items():
            if column in _IDENTITY_COLUMNS:
                continue
            metric_key = FINAL_METRIC_KEYS.get(column)
            number = _finite_number(value)
            if metric_key is None:
                if number is not None:
                    raise ValueError(
                        f"unknown numeric final metric column={column!r} in {path}"
                    )
                continue
            if number is not None:
                self.client.log_metric(run_id, metric_key, number)
        self.client.set_tag(run_id, "graph_memory.final.logged", "true")

    def _log_stable_params(self, run_id: str, values: dict[str, object]) -> None:
        existing = self.client.get_run(run_id).data.params
        for key, value in values.items():
            rendered = _param_value(value)
            previous = existing.get(key)
            if previous is not None:
                if previous != rendered:
                    raise ValueError(
                        f"MLflow parameter changed within one baseline key={key}: "
                        f"existing={previous!r} new={rendered!r}"
                    )
                continue
            self.client.log_param(run_id, key, rendered)
            existing[key] = rendered


def _output_path(invocation: StageInvocation, role: str) -> Path | None:
    return next(
        (output.path for output in invocation.outputs if output.role == role),
        None,
    )


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


def _tag_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _merged_result_rows(layout: RunLayout) -> list[dict[str, str]]:
    ordered_keys: list[tuple[str, str | None]] = []
    merged: dict[tuple[str, str | None], dict[str, str]] = {}
    for kind in ("main", "path", "efficiency", "ablation"):
        path = layout.table(kind)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            method = row.get("Method")
            if not method:
                raise ValueError(f"aggregate result row has no Method: {path}")
            variant_value = row.get("Variant") or None
            variant = None if variant_value == "full_rgcn" else variant_value
            key = (method, variant)
            if key not in merged:
                ordered_keys.append(key)
                merged[key] = {
                    "Method": method,
                    **({"Variant": variant} if variant is not None else {}),
                }
            target = merged[key]
            for column, value in row.items():
                if column in _IDENTITY_COLUMNS or value is None:
                    continue
                previous = target.get(column)
                if previous is not None and previous != value:
                    raise ValueError(
                        f"conflicting aggregate value baseline={key} column={column}: "
                        f"{previous!r} != {value!r}"
                    )
                target[column] = value
    return [merged[key] for key in ordered_keys]


def _render_result_table(rows: list[dict[str, str]]) -> str:
    columns: list[str] = ["Method"]
    if any("Variant" in row for row in rows):
        columns.append("Variant")
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_markdown_cell(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join(["Final baseline results", "", header, separator, *body])


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


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
    "BaselineIdentity",
    "FINAL_METRIC_KEYS",
    "MAX_TRACKED_ARTIFACT_BYTES",
    "TrackingAdapter",
]
