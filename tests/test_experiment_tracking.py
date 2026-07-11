from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.execution import execute_experiment
from graph_memory.experiment import execution as execution_module
from graph_memory.experiment.layout import RunLayout
from graph_memory.experiment.service import initialize_experiment
from graph_memory.experiment.state import read_stage_summary, summary_path_for
from graph_memory.experiment.status import inspect_invocation_status
from graph_memory.experiment.tracking import TrackingAdapter

ROOT = Path(__file__).resolve().parents[1]


def _config(
    tmp_path: Path,
    name: str,
    *,
    methods: str = "[bm25]",
    stages_to: str = "prepare",
):
    source = (ROOT / "tests/fixtures/hotpotqa_smoke.json").as_posix()
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(
            config_name="config",
            overrides=[
                f"name={name}",
                "profile=smoke",
                f"methods={methods}",
                "device=cpu",
                f"stages.to={stages_to}",
                f"tracking.database={(tmp_path / 'mlflow.db').as_posix()}",
                f"tracking.artifact_root={(tmp_path / 'artifacts').as_posix()}",
                *[
                    f"dataset.splits.{split}.source={source}"
                    for split in ("train", "dev", "test")
                ],
                *[
                    override
                    for split in ("train", "dev", "test")
                    for override in (
                        f"dataset.splits.{split}.offset=0",
                        f"dataset.splits.{split}.capacity=1",
                    )
                ],
            ],
        )
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )


def test_strict_mlflow_parent_children_curated_artifacts_and_cache_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f"tracking-{tmp_path.name}"
    config = _config(tmp_path, name)
    layout = RunLayout(ROOT, name)
    try:
        initialized = initialize_experiment(config, layout=layout)
        adapter = TrackingAdapter(config)
        monkeypatch.setattr(
            execution_module, "TrackingAdapter", lambda _config: adapter
        )
        first = execute_experiment(initialized)
        assert len(first.resume.skipped) == 0
        assert all(row.state == "complete" for row in first.status)

        runs = adapter.client.search_runs([adapter.experiment_id])
        assert len(runs) == 4
        parent = next(
            run for run in runs if run.data.tags["graph_memory.run_kind"] == "parent"
        )
        children = [
            run for run in runs if run.data.tags["graph_memory.run_kind"] == "stage"
        ]
        assert len(children) == 3
        assert {run.data.tags["mlflow.parentRunId"] for run in children} == {
            parent.info.run_id
        }
        child = children[0]
        assert child.data.params["artifact.inputs.kind"] == "file"
        assert child.data.tags["artifact.inputs.upload"] == "prohibited"
        assert adapter.client.list_artifacts(child.info.run_id, "stage_summary")
        assert not adapter.client.list_artifacts(child.info.run_id, "inputs")

        reopened = initialize_experiment(config, layout=layout)
        second = execute_experiment(reopened)
        assert len(second.resume.skipped) == 3
        assert second.state.mlflow_parent_run_id == parent.info.run_id
        assert len(adapter.client.search_runs([adapter.experiment_id])) == 4
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_tracking_failure_after_output_marks_summary_failed_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f"tracking-failure-{tmp_path.name}"
    config = _config(tmp_path, name)
    layout = RunLayout(ROOT, name)

    class FailingTracking:
        def start_parent(self, *, existing_run_id=None):
            return existing_run_id or "parent"

        def start_child(self, parent_run_id, invocation):
            return f"child-{invocation.identifier}"

        def log_stage(self, *args, **kwargs):
            raise RuntimeError("tracking write failed")

        def finish(self, run_id, *, status):
            pass

    try:
        initialized = initialize_experiment(config, layout=layout)
        failing = FailingTracking()
        monkeypatch.setattr(
            execution_module,
            "TrackingAdapter",
            lambda _config: failing,
        )
        with pytest.raises(RuntimeError, match="tracking write failed"):
            execute_experiment(initialized)
        first = initialized.plan.invocations[0]
        summary = read_stage_summary(summary_path_for(first))
        assert summary.status == "failed"
        assert summary.error and summary.error.type == "RuntimeError"
        assert inspect_invocation_status(first).state == "stale"
        assert not initialized.plan.invocations[1].primary_output.path.exists()
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_direct_stage_script_does_not_create_orphan_mlflow_run(tmp_path: Path) -> None:
    name = f"direct-no-orphan-{tmp_path.name}"
    config = _config(tmp_path, name)
    layout = RunLayout(ROOT, name)
    try:
        initialized = initialize_experiment(config, layout=layout)
        adapter = TrackingAdapter(config)
        invocation = initialized.plan.invocations[0]
        completed = subprocess.run(
            invocation.argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert adapter.client.search_runs([adapter.experiment_id]) == []
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_tuning_selection_logs_selected_params_and_best_metrics(tmp_path: Path) -> None:
    name = f"tracking-tune-{tmp_path.name}"
    config = _config(
        tmp_path,
        name,
        methods="[bm25_graph_rerank]",
        stages_to="tune",
    )
    layout = RunLayout(ROOT, name)
    try:
        initialized = initialize_experiment(config, layout=layout)
        invocation = next(
            item for item in initialized.plan.invocations if item.stage == "tune"
        )
        selected = next(
            output.path
            for output in invocation.outputs
            if output.role == "selected_config"
        )
        candidates = next(
            output.path
            for output in invocation.outputs
            if output.role == "candidate_table"
        )
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text('{"lambda_query":0.5}\n', encoding="utf-8")
        candidates.write_text(
            json.dumps(
                [
                    {
                        "config": {"lambda_query": 0.5},
                        "objective": 0.75,
                        "full_support_recall_at_10": 0.5,
                    }
                ]
            ),
            encoding="utf-8",
        )
        adapter = TrackingAdapter(config)
        parent = adapter.start_parent()
        child = adapter.start_child(parent, invocation)

        adapter._log_tuning_selection(child, invocation)

        run = adapter.client.get_run(child)
        assert run.data.params["tune.selected.lambda_query"] == "0.5"
        assert run.data.metrics["tune.best.objective"] == 0.75
        assert run.data.metrics["tune.best.full_support_recall_at_10"] == 0.5
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)
