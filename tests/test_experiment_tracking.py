from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

from graph_memory.experiment import execution as execution_module
from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.execution import (
    _baseline_identities,
    _tracking_owner,
    execute_experiment,
)
from graph_memory.experiment.layout import MultirunIdentity, RunLayout
from graph_memory.experiment.service import initialize_experiment
from graph_memory.experiment.state import read_stage_summary
from graph_memory.experiment.state import stage_lifecycle
from graph_memory.experiment.status import inspect_invocation_status
from graph_memory.experiment.tracking import (
    BaselineIdentity,
    FINAL_METRIC_KEYS,
    TrackingAdapter,
)
from graph_memory.registry.retrieval import RetrievalMethodId

ROOT = Path(__file__).resolve().parents[1]


def _config(
    tmp_path: Path,
    name: str,
    *,
    methods: str = "[bm25]",
    stages_from: str | None = None,
    stages_to: str = "prepare",
    extra: list[str] | None = None,
):
    source = (ROOT / "tests/fixtures/hotpotqa_smoke.json").as_posix()
    overrides = [
        f"name={name}",
        "dataset=hotpotqa",
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
        *(extra or []),
    ]
    if stages_from is not None:
        overrides.append(f"stages.from={stages_from}")
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(config_name="config", overrides=overrides)
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )


def _runs(adapter: TrackingAdapter):
    return adapter.client.search_runs([adapter.experiment_id])


def _materialize_fake_success(invocation) -> None:
    metric_rows: list[dict[str, str]] = []
    ablation_rows: list[dict[str, str]] = []
    if invocation.stage == "aggregate":
        for artifact in invocation.inputs:
            if (
                artifact.role not in {"metrics", "ablation_metrics"}
                or not artifact.path.is_file()
            ):
                continue
            with artifact.path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                if artifact.role == "metrics":
                    metric_rows.append(rows[0])
                else:
                    ablation_rows.append(
                        {
                            **rows[0],
                            "Variant": artifact.path.parents[1].name,
                        }
                    )
    for output in invocation.outputs:
        output.path.parent.mkdir(parents=True, exist_ok=True)
        if output.kind == "directory":
            output.path.mkdir(parents=True, exist_ok=True)
        elif output.role == "metrics":
            output.path.write_text(
                f"Method,Recall@10\n{invocation.method.value},0.5\n",
                encoding="utf-8",
            )
        elif output.role == "main_table":
            output.path.write_text(
                "Method,Recall@10\n"
                + "".join(
                    f"{row['Method']},{row['Recall@10']}\n" for row in metric_rows
                ),
                encoding="utf-8",
            )
        elif output.role == "path_table":
            output.path.write_text(
                "Method,Path Recall@10\n"
                + "".join(f"{row['Method']},N/A\n" for row in metric_rows),
                encoding="utf-8",
            )
        elif output.role == "efficiency_table":
            output.path.write_text(
                "Method,Memory Size\n"
                + "".join(f"{row['Method']},1\n" for row in metric_rows),
                encoding="utf-8",
            )
        elif output.role == "ablation_table":
            output.path.write_text(
                "Method,Variant,Recall@10\n"
                + "".join(
                    f"{row['Method']},{row['Variant']},{row['Recall@10']}\n"
                    for row in ablation_rows
                ),
                encoding="utf-8",
            )
        elif output.role == "train_metrics":
            output.path.write_text(
                json.dumps(
                    {
                        "epoch": 0,
                        "train_loss": 0.5,
                        "dev_loss": 0.4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            output.path.write_text("[]\n", encoding="utf-8")
    with stage_lifecycle(invocation):
        pass


def _completed(invocation, returncode: int = 0):
    return subprocess.CompletedProcess(invocation.argv, returncode)


def test_full_job_has_one_baseline_child_and_curated_parent_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f"tracking-{tmp_path.name}"
    config = _config(tmp_path, name, stages_to="aggregate")
    layout = RunLayout(ROOT, name)
    try:
        initialized = initialize_experiment(config, layout=layout)
        adapter = TrackingAdapter(config, layout)
        monkeypatch.setattr(
            execution_module,
            "TrackingAdapter",
            lambda _config, _layout: adapter,
        )

        first = execute_experiment(initialized)

        assert len(first.resume.skipped) == 0
        assert all(row.state == "complete" for row in first.status)
        runs = _runs(adapter)
        assert len(runs) == 2
        parent = next(
            run for run in runs if run.data.tags["graph_memory.run_kind"] == "parent"
        )
        child = next(
            run for run in runs if run.data.tags["graph_memory.run_kind"] == "baseline"
        )
        assert child.data.tags[MLFLOW_PARENT_RUN_ID] == parent.info.run_id
        assert child.data.tags["graph_memory.method"] == "bm25"
        assert "graph_memory.stage.prepare_train.status" not in child.data.tags
        assert "graph_memory.stage.graphs_train.status" not in child.data.tags
        assert "graph_memory.stage.aggregate_aggregate.status" not in child.data.tags
        assert parent.data.metrics == {}
        assert not any(key.startswith("method_configs.") for key in parent.data.params)
        assert not any(key.startswith("search_spaces.") for key in parent.data.params)
        assert set(parent.data.params) >= {
            "name",
            "dataset",
            "profile",
            "seed",
            "device",
            "methods",
            "top_k",
            "stages.from",
            "stages.to",
            "cache.enabled",
            "ablation.enable",
            "ablation.variants",
        }
        assert "| Method | Recall@2" in parent.data.tags["mlflow.note.content"]
        assert all(key.startswith("final.") for key in child.data.metrics)
        assert "final.recall_at_10" in child.data.metrics
        assert adapter.client.list_artifacts(parent.info.run_id, "config")
        assert adapter.client.list_artifacts(parent.info.run_id, "results")
        assert adapter.client.list_artifacts(parent.info.run_id, "workflow")
        assert adapter.client.list_artifacts(child.info.run_id, "config")
        assert adapter.client.list_artifacts(child.info.run_id, "evaluation")
        assert adapter.client.list_artifacts(child.info.run_id, "workflow")
        assert not adapter.client.list_artifacts(child.info.run_id, "predictions")
        assert (
            child.data.tags["artifact.retrieve_bm25.predictions.upload"] == "prohibited"
        )

        reopened = initialize_experiment(config, layout=layout)
        second = execute_experiment(reopened)
        assert len(second.resume.skipped) == len(initialized.plan.invocations)
        assert len(_runs(adapter)) == 2
        assert (
            len(
                adapter.client.get_metric_history(
                    child.info.run_id,
                    "final.recall_at_10",
                )
            )
            == 1
        )
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_tracking_failure_marks_local_attempt_failed_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = f"tracking-failure-{tmp_path.name}"
    config = _config(tmp_path, name)
    layout = RunLayout(ROOT, name)

    class FailingTracking:
        def start_parent(self, *, existing_run_id=None):
            return existing_run_id or "parent"

        def log_parent_stage(self, *args, **kwargs):
            raise RuntimeError("tracking write failed")

        def finish(self, run_id, *, status):
            pass

    try:
        initialized = initialize_experiment(config, layout=layout)
        monkeypatch.setattr(
            execution_module,
            "TrackingAdapter",
            lambda _config, _layout: FailingTracking(),
        )
        with pytest.raises(RuntimeError, match="tracking write failed"):
            execute_experiment(initialized)
        first = initialized.plan.invocations[0]
        summary = read_stage_summary(first.summary_path)
        assert summary.status == "failed"
        assert summary.mlflow_child_run_id is None
        assert summary.error and summary.error.type == "RuntimeError"
        assert inspect_invocation_status(first).state == "stale"
        assert not initialized.plan.invocations[1].primary_output.path.exists()
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_shared_stage_failure_fails_parent_without_synthetic_baseline_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "shared-failure", stages_to="aggregate")
    layout = RunLayout(tmp_path, config.name)
    initialized = initialize_experiment(config, layout=layout)
    adapter = TrackingAdapter(config, layout)
    monkeypatch.setattr(
        execution_module,
        "TrackingAdapter",
        lambda _config, _layout: adapter,
    )

    def fail_prepare(invocation, _initialized):
        assert invocation.stage == "prepare"
        return _completed(invocation, 1)

    monkeypatch.setattr(execution_module, "_run_invocation", fail_prepare)
    with pytest.raises(subprocess.CalledProcessError):
        execute_experiment(initialized)

    runs = _runs(adapter)
    assert len(runs) == 1
    assert runs[0].data.tags["graph_memory.run_kind"] == "parent"
    assert runs[0].info.status == "FAILED"


def test_method_failure_and_interrupted_resume_reuse_the_owning_baseline_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "method-failure", stages_to="aggregate")
    layout = RunLayout(tmp_path, config.name)
    initialized = initialize_experiment(config, layout=layout)
    adapter = TrackingAdapter(config, layout)
    monkeypatch.setattr(
        execution_module,
        "TrackingAdapter",
        lambda _config, _layout: adapter,
    )

    def fail_retrieve(invocation, _initialized):
        if invocation.stage == "retrieve":
            try:
                with stage_lifecycle(invocation):
                    raise RuntimeError("retrieval failed")
            except RuntimeError:
                pass
            return _completed(invocation, 1)
        _materialize_fake_success(invocation)
        return _completed(invocation)

    monkeypatch.setattr(execution_module, "_run_invocation", fail_retrieve)
    with pytest.raises(subprocess.CalledProcessError):
        execute_experiment(initialized)

    runs = _runs(adapter)
    parent = next(
        run for run in runs if run.data.tags["graph_memory.run_kind"] == "parent"
    )
    child = next(
        run for run in runs if run.data.tags["graph_memory.run_kind"] == "baseline"
    )
    assert parent.info.status == "FAILED"
    assert child.info.status == "FAILED"
    assert child.data.tags["graph_memory.stage.retrieve_bm25.status"] == "failed"

    def succeed(invocation, _initialized):
        _materialize_fake_success(invocation)
        return _completed(invocation)

    monkeypatch.setattr(execution_module, "_run_invocation", succeed)
    resumed = initialize_experiment(config, layout=layout)
    result = execute_experiment(resumed)

    assert result.resume.first_pending is not None
    assert result.resume.first_pending.stage == "retrieve"
    assert len(_runs(adapter)) == 2
    assert adapter.client.get_run(parent.info.run_id).info.status == "FINISHED"
    assert adapter.client.get_run(child.info.run_id).info.status == "FINISHED"
    assert (
        len(adapter.client.get_metric_history(child.info.run_id, "final.recall_at_10"))
        == 1
    )


def test_cache_disabled_repeated_attempts_keep_one_child_and_one_final_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        "cache-disabled",
        stages_to="aggregate",
        extra=["cache.enabled=false"],
    )
    layout = RunLayout(tmp_path, config.name)
    adapter = TrackingAdapter(config, layout)
    monkeypatch.setattr(
        execution_module,
        "TrackingAdapter",
        lambda _config, _layout: adapter,
    )

    def succeed(invocation, _initialized):
        _materialize_fake_success(invocation)
        return _completed(invocation)

    monkeypatch.setattr(execution_module, "_run_invocation", succeed)
    first = initialize_experiment(config, layout=layout)
    execute_experiment(first)
    second = initialize_experiment(config, layout=layout)
    result = execute_experiment(second)

    assert result.resume.skipped == ()
    runs = _runs(adapter)
    assert len(runs) == 2
    child = next(
        run for run in runs if run.data.tags["graph_memory.run_kind"] == "baseline"
    )
    assert (
        len(adapter.client.get_metric_history(child.info.run_id, "final.recall_at_10"))
        == 1
    )
    retrieve = next(
        item for item in second.plan.invocations if item.stage == "retrieve"
    )
    assert read_stage_summary(retrieve.summary_path).attempt == 2


def test_fully_cached_hidden_dependency_populates_one_selected_baseline_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        "cached-hidden-dependency",
        methods="[dense_ft_rgcn_graph_retriever]",
        stages_to="train",
    )
    layout = RunLayout(tmp_path, config.name)
    initialized = initialize_experiment(config, layout=layout)
    for invocation in initialized.plan.invocations:
        _materialize_fake_success(invocation)
    adapter = TrackingAdapter(config, layout)
    monkeypatch.setattr(
        execution_module,
        "TrackingAdapter",
        lambda _config, _layout: adapter,
    )

    result = execute_experiment(initialized)

    assert result.resume.invocations == ()
    runs = _runs(adapter)
    assert len(runs) == 2
    child = next(
        run for run in runs if run.data.tags["graph_memory.run_kind"] == "baseline"
    )
    assert child.data.tags["graph_memory.method"] == "dense_ft_rgcn_graph_retriever"
    assert adapter.client.list_artifacts(child.info.run_id, "dependencies/dense_ft")
    assert not any(
        run.data.tags.get("graph_memory.method") == "dense_ft" for run in runs
    )


def test_direct_stage_script_does_not_create_orphan_mlflow_run(tmp_path: Path) -> None:
    name = f"direct-no-orphan-{tmp_path.name}"
    config = _config(tmp_path, name)
    layout = RunLayout(ROOT, name)
    try:
        initialized = initialize_experiment(config, layout=layout)
        adapter = TrackingAdapter(config, layout)
        invocation = initialized.plan.invocations[0]
        completed = subprocess.run(
            invocation.argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert _runs(adapter) == []
    finally:
        shutil.rmtree(layout.run_dir, ignore_errors=True)


def test_parent_projection_and_multirun_identity_are_concise(tmp_path: Path) -> None:
    config = _config(tmp_path, "parent-projection")
    layout = RunLayout(
        tmp_path,
        config.name,
        identity=MultirunIdentity(job_num=2, suffix="num_layers=4"),
    )
    initialized = initialize_experiment(
        config,
        layout=layout,
        overrides=(
            "name=parent-projection",
            "method_configs.dense_rgcn_graph_retriever.train.model.num_layers=4",
        ),
    )
    adapter = TrackingAdapter(config, layout)

    parent_id = adapter.start_parent()
    parent = adapter.client.get_run(parent_id)

    assert parent.data.tags[MLFLOW_RUN_NAME] == "parent-projection/2_num_layers=4"
    assert parent.data.params["multirun.job_num"] == "2"
    assert parent.data.params["multirun.suffix"] == "num_layers=4"
    assert "method_configs.dense_rgcn_graph_retriever" not in json.dumps(
        parent.data.params
    )
    assert adapter.client.list_artifacts(parent_id, "config")
    assert initialized.overrides[-1].endswith("num_layers=4")


def test_final_metric_mapping_omits_na_nonfinite_and_rejects_unknown_numeric(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "metric-hygiene")
    layout = RunLayout(tmp_path, config.name)
    initialize_experiment(config, layout=layout)
    metric = layout.metric(RetrievalMethodId.BM25)
    metric.parent.mkdir(parents=True, exist_ok=True)
    metric.write_text(
        "Method,Recall@5,Path Recall@10,Edge Recall@10\nbm25,0.5,N/A,nan\n",
        encoding="utf-8",
    )
    adapter = TrackingAdapter(config, layout)
    parent = adapter.start_parent()
    identity = BaselineIdentity(RetrievalMethodId.BM25)
    child = adapter.get_or_create_baseline_child(parent, identity)

    adapter.finalize_baseline(child, identity)

    run = adapter.client.get_run(child)
    assert run.data.metrics == {"final.recall_at_5": 0.5}
    assert set(FINAL_METRIC_KEYS.values()) >= {"final.recall_at_5"}
    adapter.client.delete_tag(child, "graph_memory.final.logged")
    metric.write_text(
        "Method,Recall@5,Unexpected Count\nbm25,0.5,7\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown numeric final metric"):
        adapter.finalize_baseline(child, identity)


def test_training_projection_logs_only_finite_flat_epoch_series_and_method_params(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        "train-series",
        methods="[dense_rgcn_graph_retriever]",
        stages_to="train",
    )
    layout = RunLayout(tmp_path, config.name)
    initialized = initialize_experiment(config, layout=layout)
    invocation = next(
        item for item in initialized.plan.invocations if item.stage == "train"
    )
    metrics = next(
        output.path for output in invocation.outputs if output.role == "train_metrics"
    )
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        json.dumps(
            {
                "epoch": 1,
                "train_loss": 0.4,
                "dev_loss": 0.3,
                "nested": {"bad": 1},
                "flag": True,
                "label": "ignored",
                "nan": float("nan"),
                "infinite": float("inf"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = TrackingAdapter(config, layout)
    parent = adapter.start_parent()
    identity = BaselineIdentity(RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER)
    child = adapter.get_or_create_baseline_child(parent, identity)

    adapter._log_effective_method_parameters(child, invocation)
    adapter._log_training_metrics(child, invocation)

    run = adapter.client.get_run(child)
    assert run.data.metrics == {"train.train_loss": 0.4, "train.dev_loss": 0.3}
    assert run.data.params["train.trainer.epochs"]
    assert not any(key.startswith("stage_config.") for key in run.data.params)
    assert len(adapter.client.get_metric_history(child, "train.train_loss")) == 1
    assert adapter.client.get_metric_history(child, "train.train_loss")[0].step == 1


def test_duplicate_baseline_children_fail_strictly(tmp_path: Path) -> None:
    config = _config(tmp_path, "duplicate-child")
    layout = RunLayout(tmp_path, config.name)
    initialize_experiment(config, layout=layout)
    adapter = TrackingAdapter(config, layout)
    parent = adapter.start_parent()
    tags = {
        MLFLOW_PARENT_RUN_ID: parent,
        MLFLOW_RUN_NAME: "bm25",
        "graph_memory.run_kind": "baseline",
        "graph_memory.method": "bm25",
        "graph_memory.variant": "ordinary",
    }
    adapter.client.create_run(adapter.experiment_id, tags=tags)
    adapter.client.create_run(adapter.experiment_id, tags=tags)

    with pytest.raises(ValueError, match="duplicate MLflow baseline children"):
        adapter.get_or_create_baseline_child(
            parent,
            BaselineIdentity(RetrievalMethodId.BM25),
        )


def test_hidden_explicit_dependency_ablation_and_stage_bounds_have_stable_owners(
    tmp_path: Path,
) -> None:
    hidden_config = _config(
        tmp_path,
        "hidden-dependency",
        methods="[dense_ft_rgcn_graph_retriever]",
        stages_to="train",
    )
    hidden = initialize_experiment(
        hidden_config,
        layout=RunLayout(tmp_path, hidden_config.name),
    )
    dense_ft_train = next(
        item
        for item in hidden.plan.invocations
        if item.stage == "train" and item.method is RetrievalMethodId.DENSE_FT
    )
    assert _tracking_owner(hidden, dense_ft_train) == (
        BaselineIdentity(RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER),
        True,
    )

    explicit_config = _config(
        tmp_path,
        "explicit-dependency",
        methods="[dense_ft,dense_ft_rgcn_graph_retriever]",
        stages_to="train",
    )
    explicit = initialize_experiment(
        explicit_config,
        layout=RunLayout(tmp_path, explicit_config.name),
    )
    explicit_dense_ft = next(
        item
        for item in explicit.plan.invocations
        if item.stage == "train" and item.method is RetrievalMethodId.DENSE_FT
    )
    assert _tracking_owner(explicit, explicit_dense_ft) == (
        BaselineIdentity(RetrievalMethodId.DENSE_FT),
        False,
    )

    ablation_config = _config(
        tmp_path,
        "ablation-identities",
        methods="[dense_rgcn_graph_retriever]",
        stages_to="aggregate",
        extra=["ablation.enable=true", "ablation.variants=[wo_graph]"],
    )
    ablation = initialize_experiment(
        ablation_config,
        layout=RunLayout(tmp_path, ablation_config.name),
    )
    assert _baseline_identities(ablation) == (
        BaselineIdentity(RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER),
        BaselineIdentity(RetrievalMethodId.DENSE_RGCN_GRAPH_RETRIEVER, "wo_graph"),
    )

    bounded_config = _config(tmp_path, "shared-only", stages_to="prepare")
    bounded = initialize_experiment(
        bounded_config,
        layout=RunLayout(tmp_path, bounded_config.name),
    )
    assert _baseline_identities(bounded) == ()


def test_fresh_six_evidence_baseline_tracking_preserves_existing_store_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    methods = (
        "[bm25,dense,dense_ft,graphrag,dense_rgcn_graph_retriever,"
        "dense_ft_rgcn_graph_retriever]"
    )
    config = _config(
        tmp_path,
        "accept-six-evidence",
        methods=methods,
        stages_to="aggregate",
    )
    layout = RunLayout(tmp_path, config.name)
    initialized = initialize_experiment(config, layout=layout)
    before = TrackingAdapter(config, layout)
    sentinel_id = before.client.create_run(
        before.experiment_id,
        tags={MLFLOW_RUN_NAME: "historical-sentinel", "sentinel": "untouched"},
    ).info.run_id
    before.client.log_metric(sentinel_id, "historical.metric", 7.0)
    before.client.set_terminated(sentinel_id, status="FINISHED")

    def succeed(invocation, _initialized):
        _materialize_fake_success(invocation)
        return _completed(invocation)

    monkeypatch.setattr(execution_module, "_run_invocation", succeed)
    execute_experiment(initialized)

    adapter = TrackingAdapter(config, layout)
    runs = _runs(adapter)
    parents = [
        run for run in runs if run.data.tags.get("graph_memory.run_kind") == "parent"
    ]
    children = [
        run for run in runs if run.data.tags.get("graph_memory.run_kind") == "baseline"
    ]
    assert len(parents) == 1
    assert len(children) == 6
    parent = parents[0]
    assert parent.data.metrics == {}
    note = parent.data.tags["mlflow.note.content"]
    assert all(f"| {method} |" in note for method in config.methods)
    assert adapter.client.list_artifacts(parent.info.run_id, "config")
    assert adapter.client.list_artifacts(parent.info.run_id, "results")
    assert adapter.client.list_artifacts(parent.info.run_id, "workflow")
    assert all("final.recall_at_10" in child.data.metrics for child in children)
    trainable = {
        "dense_rgcn_graph_retriever",
        "dense_ft",
        "dense_ft_rgcn_graph_retriever",
    }
    assert all(
        "train.train_loss" in child.data.metrics
        for child in children
        if child.data.tags["graph_memory.method"] in trainable
    )
    assert not list(layout.run_dir.rglob("*.png"))
    assert not list(layout.run_dir.rglob("*.svg"))
    sentinel = adapter.client.get_run(sentinel_id)
    assert sentinel.info.status == "FINISHED"
    assert sentinel.data.tags["sentinel"] == "untouched"
    assert sentinel.data.metrics == {"historical.metric": 7.0}
    assert config.tracking.database == (tmp_path / "mlflow.db").resolve()
    assert config.tracking.artifact_root == (tmp_path / "artifacts").resolve()


def test_fresh_three_job_rgcn_layer_sweep_has_concise_parents_and_comparable_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "accept-rgcn-layers"

    def succeed(invocation, _initialized):
        _materialize_fake_success(invocation)
        return _completed(invocation)

    monkeypatch.setattr(execution_module, "_run_invocation", succeed)
    configs = []
    layouts = []
    for job_num, num_layers in enumerate((2, 3, 4)):
        config = _config(
            tmp_path,
            name,
            methods="[dense_rgcn_graph_retriever]",
            stages_to="aggregate",
            extra=[
                "method_configs.dense_rgcn_graph_retriever.train.model.num_layers="
                f"{num_layers}"
            ],
        )
        layout = RunLayout(
            tmp_path,
            name,
            identity=MultirunIdentity(
                job_num=job_num,
                suffix=f"num_layers={num_layers}",
            ),
        )
        execute_experiment(
            initialize_experiment(
                config,
                layout=layout,
                overrides=(
                    f"name={name}",
                    "methods=[dense_rgcn_graph_retriever]",
                    "method_configs.dense_rgcn_graph_retriever.train.model.num_layers="
                    f"{num_layers}",
                ),
            )
        )
        configs.append(config)
        layouts.append(layout)

    adapter = TrackingAdapter(configs[0], layouts[0])
    runs = _runs(adapter)
    parents = [
        run for run in runs if run.data.tags.get("graph_memory.run_kind") == "parent"
    ]
    children = [
        run for run in runs if run.data.tags.get("graph_memory.run_kind") == "baseline"
    ]
    assert [layout.run_dir.name for layout in layouts] == [
        "0_num_layers=2",
        "1_num_layers=3",
        "2_num_layers=4",
    ]
    assert {run.data.tags[MLFLOW_RUN_NAME] for run in parents} == {
        f"{name}/0_num_layers=2",
        f"{name}/1_num_layers=3",
        f"{name}/2_num_layers=4",
    }
    assert len(parents) == 3
    assert len(children) == 3
    assert {child.data.params["train.model.num_layers"] for child in children} == {
        "2",
        "3",
        "4",
    }
    assert all(set(child.data.metrics) >= {"final.recall_at_10"} for child in children)
    assert len({child.data.tags["mlflow.parentRunId"] for child in children}) == 3


def test_fresh_ablation_workflow_tracks_ordinary_and_variant_baselines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        "accept-ablation",
        methods="[dense_rgcn_graph_retriever]",
        stages_to="aggregate",
        extra=["ablation.enable=true", "ablation.variants=[wo_graph]"],
    )
    layout = RunLayout(tmp_path, config.name)
    initialized = initialize_experiment(config, layout=layout)

    def succeed(invocation, _initialized):
        _materialize_fake_success(invocation)
        return _completed(invocation)

    monkeypatch.setattr(execution_module, "_run_invocation", succeed)
    execute_experiment(initialized)

    adapter = TrackingAdapter(config, layout)
    runs = _runs(adapter)
    parent = next(
        run for run in runs if run.data.tags.get("graph_memory.run_kind") == "parent"
    )
    children = [
        run for run in runs if run.data.tags.get("graph_memory.run_kind") == "baseline"
    ]
    assert {child.data.tags["graph_memory.variant"] for child in children} == {
        "ordinary",
        "wo_graph",
    }
    assert all("final.recall_at_10" in child.data.metrics for child in children)
    assert (
        "| dense_rgcn_graph_retriever | wo_graph |"
        in parent.data.tags["mlflow.note.content"]
    )
