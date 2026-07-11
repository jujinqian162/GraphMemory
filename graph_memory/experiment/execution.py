from __future__ import annotations

import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone

from graph_memory.experiment.persistence import write_yaml_atomic
from graph_memory.experiment.planning import StageInvocation, format_invocation
from graph_memory.experiment.resume import ResumeDecision, resume_plan
from graph_memory.experiment.service import InitializedExperiment
from graph_memory.experiment.state import (
    ErrorRecord,
    RunState,
    StageRunSummary,
    read_stage_summary,
    summary_path_for,
    update_run_state,
    write_stage_summary,
)
from graph_memory.experiment.status import StatusRow, inspect_plan_status
from graph_memory.experiment.tracking import TrackingAdapter


@dataclass(frozen=True)
class ExecutionResult:
    state: RunState
    resume: ResumeDecision
    status: tuple[StatusRow, ...]


def execute_experiment(
    initialized: InitializedExperiment,
) -> ExecutionResult:
    adapter = TrackingAdapter(initialized.config)
    parent_run_id = adapter.start_parent(
        existing_run_id=initialized.state.mlflow_parent_run_id
    )
    state = update_run_state(
        initialized.layout.run_state,
        initialized.state,
        mlflow_parent_run_id=parent_run_id,
    )
    decision = resume_plan(
        initialized.plan,
        cache_enabled=initialized.config.cache.enabled,
    )
    for index, invocation in enumerate(decision.invocations, start=1):
        if invocation.stage == "aggregate" and initialized.plan.ablation_selections:
            _write_ablation_index(initialized)
        print(format_invocation(invocation, index=index), flush=True)
        child_run_id = adapter.start_child(parent_run_id, invocation)
        try:
            completed = subprocess.run(
                invocation.argv,
                cwd=initialized.layout.repository_root,
                check=False,
            )
            summary = _summary_with_child(invocation, child_run_id)
            if completed.returncode != 0:
                if summary is not None:
                    adapter.log_stage(
                        child_run_id,
                        invocation,
                        summary,
                        resolved_config=initialized.layout.resolved_config,
                        overrides=initialized.layout.overrides,
                    )
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    invocation.argv,
                )
            if summary is None:
                raise RuntimeError(
                    f"stage={invocation.identifier} exited successfully without a typed summary"
                )
            if summary.status != "success":
                raise RuntimeError(
                    f"stage={invocation.identifier} exited successfully with summary status={summary.status}"
                )
            adapter.log_stage(
                child_run_id,
                invocation,
                summary,
                resolved_config=initialized.layout.resolved_config,
                overrides=initialized.layout.overrides,
            )
            adapter.finish(child_run_id, status="FINISHED")
        except BaseException as error:
            _mark_tracking_or_execution_failure(invocation, child_run_id, error)
            try:
                adapter.finish(child_run_id, status="FAILED")
            finally:
                adapter.finish(parent_run_id, status="FAILED")
            raise
    adapter.finish(parent_run_id, status="FINISHED")
    return ExecutionResult(
        state=state,
        resume=decision,
        status=inspect_plan_status(initialized.plan),
    )


def _summary_with_child(
    invocation: StageInvocation,
    child_run_id: str,
) -> StageRunSummary | None:
    path = summary_path_for(invocation)
    if not path.is_file():
        return None
    summary = read_stage_summary(path).model_copy(
        update={"mlflow_child_run_id": child_run_id}
    )
    write_stage_summary(path, summary)
    return summary


def _mark_tracking_or_execution_failure(
    invocation: StageInvocation,
    child_run_id: str,
    error: BaseException,
) -> None:
    path = summary_path_for(invocation)
    if not path.is_file():
        return
    try:
        summary = read_stage_summary(path)
    except (OSError, ValueError):
        return
    failed = summary.model_copy(
        update={
            "status": "failed",
            "ended_at": datetime.now(timezone.utc),
            "error": ErrorRecord(
                type=type(error).__name__,
                message=str(error),
                traceback="".join(traceback.format_exception(error)),
            ),
            "mlflow_child_run_id": child_run_id,
        }
    )
    write_stage_summary(path, failed)


def _write_ablation_index(initialized: InitializedExperiment) -> None:
    metrics: list[dict[str, str]] = []
    for selection in initialized.plan.ablation_selections:
        path = (
            initialized.layout.metric(selection.method)
            if selection.variant == "full_rgcn"
            else initialized.layout.metric(
                selection.method,
                variant=selection.variant,
            )
        )
        metrics.append(
            {
                "method": selection.method.value,
                "variant": selection.variant,
                "metrics_path": str(path.resolve()),
            }
        )
    write_yaml_atomic(
        initialized.layout.ablation_metrics_index,
        {"metrics": metrics},
    )


__all__ = ["ExecutionResult", "execute_experiment"]
