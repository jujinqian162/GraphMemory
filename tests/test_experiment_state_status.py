from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from graph_memory.experiment.config import (
    resolve_experiment_config,
    validate_composed_config,
)
from graph_memory.experiment.layout import MultirunIdentity, RunLayout
from graph_memory.experiment.persistence import (
    read_yaml,
    write_yaml_atomic,
)
from graph_memory.experiment.planning import WorkflowPlanner
from graph_memory.experiment.resume import prune_completed_prefix
from graph_memory.experiment.service import initialize_experiment
from graph_memory.experiment.state import (
    FailedStageRunSummary,
    RunningStageRunSummary,
    StageRunSummary,
    SuccessfulStageRunSummary,
    create_run_state,
    read_run_state,
    read_stage_summary,
    stage_lifecycle,
    validate_run_identity,
    write_run_state,
    write_stage_summary,
)
from graph_memory.experiment.status import (
    StatusRow,
    inspect_invocation_status,
    inspect_plan_status,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (
        ROOT / "tests/fixtures/refactor_7_9/legacy_cache_ablation_contracts.json"
    ).read_text(encoding="utf-8")
)


def _resolved(*, overrides: list[str] | None = None):
    with initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base="1.3",
    ):
        composed = compose(config_name="config", overrides=overrides or [])
    return resolve_experiment_config(
        validate_composed_config(composed),
        repository_root=ROOT,
    )


def _materialize_outputs(invocation) -> None:
    for artifact in invocation.outputs:
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        if artifact.kind == "file":
            artifact.path.write_text("[]\n", encoding="utf-8")
        else:
            artifact.path.mkdir(parents=True, exist_ok=True)


def test_atomic_yaml_persistence_replaces_content_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested/state.yaml"
    write_yaml_atomic(path, {"version": 1, "value": "first"})
    write_yaml_atomic(path, {"version": 1, "value": "second"})
    assert read_yaml(path) == {"version": 1, "value": "second"}
    assert not list(path.parent.glob("*.tmp"))

    overrides = tmp_path / "overrides.yaml"
    write_yaml_atomic(overrides, ["dataset=twowiki", "profile=smoke"])
    assert read_yaml(overrides) == ["dataset=twowiki", "profile=smoke"]


def test_run_state_round_trip_and_identity_reuse_checks(tmp_path: Path) -> None:
    config = _resolved(overrides=["name=state-test", "methods=[bm25]"])
    layout = RunLayout(tmp_path, "state-test")
    plan = WorkflowPlanner(config, layout).build()
    state = create_run_state(layout=layout, config=config, plan=plan)
    write_run_state(layout.run_state, state)
    loaded = read_run_state(layout.run_state)
    assert loaded == state
    validate_run_identity(layout=layout, config=config, existing=loaded)

    changed = _resolved(overrides=["name=state-test", "methods=[dense]"])
    with pytest.raises(ValueError, match="config mismatch"):
        validate_run_identity(layout=layout, config=changed, existing=loaded)
    with pytest.raises(ValueError, match="mode mismatch"):
        validate_run_identity(
            layout=RunLayout(
                tmp_path,
                "state-test",
                identity=MultirunIdentity(job_num=0, suffix="x"),
            ),
            config=config,
            existing=loaded,
        )


def test_shared_initialization_persists_configs_and_reuses_exact_state(
    tmp_path: Path,
) -> None:
    config = _resolved(
        overrides=["name=initialized", "methods=[bm25]", "stages.to=prepare"]
    )
    layout = RunLayout(tmp_path, "initialized")
    first = initialize_experiment(
        config,
        layout=layout,
        overrides=("name=initialized", "methods=[bm25]", "stages.to=prepare"),
    )
    assert layout.resolved_config.is_file()
    assert layout.overrides.is_file()
    assert all(item.config_path.is_file() for item in first.plan.invocations)
    assert (
        read_run_state(layout.run_state).resolved_config_path
        == layout.resolved_config.resolve()
    )

    second = initialize_experiment(
        config,
        layout=layout,
        overrides=("name=initialized", "methods=[bm25]", "stages.to=prepare"),
    )
    assert second.state == first.state

    original_state = layout.run_state.read_text(encoding="utf-8")
    mismatch = _resolved(
        overrides=["name=initialized", "methods=[dense]", "stages.to=prepare"]
    )
    with pytest.raises(ValueError, match="config mismatch"):
        initialize_experiment(mismatch, layout=layout)
    assert layout.run_state.read_text(encoding="utf-8") == original_state


def test_stage_lifecycle_writes_running_success_and_failed_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _resolved(
        overrides=["name=lifecycle", "methods=[bm25]", "stages.to=prepare"]
    )
    invocation = (
        WorkflowPlanner(config, RunLayout(tmp_path, "lifecycle")).build().invocations[0]
    )
    seen: list[StageRunSummary] = []

    from graph_memory.experiment import state as state_module

    actual_write = state_module.write_stage_summary

    def record_summary(path: Path, summary: StageRunSummary) -> None:
        seen.append(summary)
        actual_write(path, summary)

    monkeypatch.setattr(state_module, "write_stage_summary", record_summary)
    _materialize_outputs(invocation)
    with stage_lifecycle(invocation) as observations:
        observations.count("examples", 3)
        observations.timing("load_seconds", 0.25)
    assert [summary.status for summary in seen] == ["running", "success"]
    summary = read_stage_summary(invocation.summary_path)
    assert summary.status == "success"
    assert summary.counts == {"examples": 3}
    assert inspect_invocation_status(invocation).state == "complete"

    with pytest.raises(RuntimeError, match="boom"):
        with stage_lifecycle(invocation):
            raise RuntimeError("boom")
    failed = read_stage_summary(invocation.summary_path)
    assert failed.status == "failed"
    assert failed.attempt == 2
    assert failed.error.type == "RuntimeError"
    assert inspect_invocation_status(invocation).state == "stale"

    running_data = seen[0].model_dump()
    with pytest.raises(ValueError, match="extra_forbidden"):
        RunningStageRunSummary.model_validate(
            {**running_data, "ended_at": summary.ended_at}
        )
    success_data = summary.model_dump()
    success_data.pop("ended_at")
    with pytest.raises(ValueError, match="ended_at"):
        SuccessfulStageRunSummary.model_validate(success_data)
    failed_data = failed.model_dump()
    failed_data.pop("error")
    with pytest.raises(ValueError, match="error"):
        FailedStageRunSummary.model_validate(failed_data)


def test_status_requires_all_outputs_matching_success_summary_and_artifact_kind(
    tmp_path: Path,
) -> None:
    config = _resolved(overrides=["name=status", "methods=[dense_ft]"])
    plan = WorkflowPlanner(config, RunLayout(tmp_path, "status")).build()
    prepare = plan.invocations[0]
    assert inspect_invocation_status(prepare).state == "missing"
    prepare.outputs[0].path.parent.mkdir(parents=True, exist_ok=True)
    prepare.outputs[0].path.write_text("[]", encoding="utf-8")
    assert inspect_invocation_status(prepare).state == "stale"
    _materialize_outputs(prepare)
    with stage_lifecycle(prepare):
        pass
    assert inspect_invocation_status(prepare).state == "complete"

    train = next(
        item for item in plan.invocations if item.identifier == "train:dense_ft"
    )
    _materialize_outputs(train)
    with stage_lifecycle(train):
        pass
    assert train.primary_output.kind == "directory"
    assert inspect_invocation_status(train).state == "complete"
    train.primary_output.path.rmdir()
    train.primary_output.path.write_text("wrong kind", encoding="utf-8")
    assert inspect_invocation_status(train).state == "stale"


def test_status_reports_ablation_aliases_without_materializing_copies(
    tmp_path: Path,
) -> None:
    config = _resolved(
        overrides=[
            "name=aliases",
            "methods=[dense_rgcn_graph_retriever]",
            "ablation.variants=[wo_graph]",
        ]
    )
    plan = WorkflowPlanner(config, RunLayout(tmp_path, "aliases")).build()
    for invocation in plan.invocations:
        if invocation.identifier in {
            "pairs:dense_rgcn_graph_retriever",
            "train:dense_rgcn_graph_retriever",
            "retrieve:dense_rgcn_graph_retriever",
            "evaluate:dense_rgcn_graph_retriever",
        }:
            _materialize_outputs(invocation)
            with stage_lifecycle(invocation):
                pass
    rows = inspect_plan_status(plan)
    aliases = [row for row in rows if row.state == "alias"]
    assert [(row.stage, row.variant) for row in aliases] == [
        ("pairs", "full_rgcn"),
        ("train", "full_rgcn"),
        ("retrieve", "full_rgcn"),
        ("evaluate", "full_rgcn"),
        ("pairs", "wo_graph"),
    ]
    assert all(not row.path.exists() for row in aliases)


@pytest.mark.parametrize("case", FIXTURE["resume_cases"])
def test_completed_prefix_resume_matches_frozen_contract(
    case: dict[str, object],
) -> None:
    config = _resolved(overrides=["name=resume", "methods=[bm25]", "stages.to=graphs"])
    plan = WorkflowPlanner(config, RunLayout(ROOT, "resume")).build()
    invocations = plan.invocations[:4]
    states = case["states"]
    assert isinstance(states, list)
    rows = tuple(
        StatusRow(
            identifier=item.identifier,
            stage=item.stage,
            state=state,
            path=item.primary_output.path,
            method=item.method,
            split=item.split,
            variant=item.variant,
            reason=None,
        )
        for item, state in zip(invocations, states, strict=True)
    )
    decision = prune_completed_prefix(invocations, rows, cache_enabled=True)
    assert len(decision.skipped) == case["skipped"]
    assert len(decision.invocations) == case["remaining"]
    expected_pending = case["first_pending"]
    assert (
        None
        if decision.first_pending is None
        else invocations.index(decision.first_pending)
    ) == expected_pending

    uncached = prune_completed_prefix(invocations, rows, cache_enabled=False)
    assert uncached.invocations == invocations
    assert uncached.skipped == ()


def test_matching_outputs_with_mismatched_summary_are_stale(tmp_path: Path) -> None:
    config = _resolved(
        overrides=["name=mismatch", "methods=[bm25]", "stages.to=prepare"]
    )
    invocation = (
        WorkflowPlanner(config, RunLayout(tmp_path, "mismatch")).build().invocations[0]
    )
    _materialize_outputs(invocation)
    with stage_lifecycle(invocation):
        pass
    summary = read_stage_summary(invocation.summary_path)
    write_stage_summary(
        invocation.summary_path,
        summary.model_copy(update={"effective_config": {"stage": "prepare"}}),
    )
    row = inspect_invocation_status(invocation)
    assert row.state == "stale"
    assert row.reason == "effective config mismatch"
