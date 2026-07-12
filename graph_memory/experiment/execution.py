from __future__ import annotations

import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone

from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.persistence import write_yaml_atomic
from graph_memory.experiment.planning import format_invocation
from graph_memory.experiment.resume import ResumeDecision, resume_plan
from graph_memory.experiment.service import InitializedExperiment
from graph_memory.experiment.state import (
    ErrorRecord,
    FailedStageRunSummary,
    RunState,
    StageRunSummary,
    read_stage_summary,
    update_run_state,
    write_stage_summary,
)
from graph_memory.experiment.status import StatusRow, inspect_plan_status
from graph_memory.experiment.tracking import BaselineIdentity, TrackingAdapter
from graph_memory.registry.retrieval import RetrievalMethodId


@dataclass(frozen=True)
class ExecutionResult:
    state: RunState
    resume: ResumeDecision
    status: tuple[StatusRow, ...]


def execute_experiment(
    initialized: InitializedExperiment,
) -> ExecutionResult:
    adapter = TrackingAdapter(initialized.config, initialized.layout)
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
    child_ids: dict[BaselineIdentity, str] = {}
    current_invocation: StageInvocation | None = None
    current_child_id: str | None = None

    def child_for(identity: BaselineIdentity) -> str:
        child_id = child_ids.get(identity)
        if child_id is None:
            child_id = adapter.get_or_create_baseline_child(parent_run_id, identity)
            child_ids[identity] = child_id
        return child_id

    try:
        for index, invocation in enumerate(decision.invocations, start=1):
            current_invocation = invocation
            owner = _tracking_owner(initialized, invocation)
            current_child_id = child_for(owner[0]) if owner is not None else None
            if invocation.stage == "aggregate" and initialized.plan.ablation_selections:
                _write_ablation_index(initialized)
            print(format_invocation(invocation, index=index), flush=True)
            completed = _run_invocation(invocation, initialized)
            summary = _summary_with_child(invocation, current_child_id)
            if completed.returncode != 0:
                if summary is not None:
                    _log_stage(
                        adapter,
                        parent_run_id,
                        current_child_id,
                        owner,
                        invocation,
                        summary,
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
            _log_stage(
                adapter,
                parent_run_id,
                current_child_id,
                owner,
                invocation,
                summary,
            )

        # Cached stages have no new subprocess event. Project their authoritative local
        # summaries into the same baseline children without changing cache decisions.
        for invocation in initialized.plan.invocations:
            if not invocation.summary_path.is_file():
                continue
            summary = read_stage_summary(invocation.summary_path)
            if summary.status != "success":
                continue
            owner = _tracking_owner(initialized, invocation)
            child_id = child_for(owner[0]) if owner is not None else None
            summary = _summary_with_child(invocation, child_id) or summary
            _log_stage(
                adapter,
                parent_run_id,
                child_id,
                owner,
                invocation,
                summary,
            )

        for identity in _baseline_identities(initialized):
            child_id = child_for(identity)
            adapter.finalize_baseline(child_id, identity)

        dense_ft = BaselineIdentity(RetrievalMethodId.DENSE_FT)
        dependent = BaselineIdentity(
            RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER
        )
        if dense_ft in child_ids and dependent in child_ids:
            adapter.reference_dependency(
                child_ids[dependent],
                dependency=dense_ft,
                dependency_run_id=child_ids[dense_ft],
            )

        adapter.finalize_parent(parent_run_id)
        for child_id in child_ids.values():
            adapter.finish(child_id, status="FINISHED")
        adapter.finish(parent_run_id, status="FINISHED")
    except BaseException as error:
        if current_invocation is not None:
            _mark_tracking_or_execution_failure(
                current_invocation,
                current_child_id,
                error,
            )
        for child_id in child_ids.values():
            try:
                adapter.finish(child_id, status="FAILED")
            except BaseException:
                pass
        try:
            adapter.finish(parent_run_id, status="FAILED")
        finally:
            raise

    return ExecutionResult(
        state=state,
        resume=decision,
        status=inspect_plan_status(initialized.plan),
    )


def _run_invocation(
    invocation: StageInvocation,
    initialized: InitializedExperiment,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        invocation.argv,
        cwd=initialized.layout.repository_root,
        check=False,
    )


def _tracking_owner(
    initialized: InitializedExperiment,
    invocation: StageInvocation,
) -> tuple[BaselineIdentity, bool] | None:
    method = invocation.method
    if method is None:
        return None
    selected = set(initialized.config.methods)
    if method in selected:
        return BaselineIdentity(method, invocation.variant), False
    if (
        method is RetrievalMethodId.DENSE_FT
        and RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER in selected
    ):
        return BaselineIdentity(RetrievalMethodId.DENSE_FT_RGCN_GRAPH_RETRIEVER), True
    raise ValueError(
        f"method-owned stage has no user-visible baseline owner: {invocation.identifier}"
    )


def _baseline_identities(
    initialized: InitializedExperiment,
) -> tuple[BaselineIdentity, ...]:
    identities: list[BaselineIdentity] = []
    for invocation in initialized.plan.invocations:
        owner = _tracking_owner(initialized, invocation)
        if owner is not None and owner[0] not in identities:
            identities.append(owner[0])
    if any(item.stage == "aggregate" for item in initialized.plan.invocations):
        for method in initialized.config.methods:
            identity = BaselineIdentity(method)
            if identity not in identities:
                identities.append(identity)
        for selection in initialized.plan.ablation_selections:
            identity = BaselineIdentity(
                selection.method,
                None if selection.variant == "full_rgcn" else selection.variant,
            )
            if identity not in identities:
                identities.append(identity)
    return tuple(identities)


def _log_stage(
    adapter: TrackingAdapter,
    parent_run_id: str,
    child_run_id: str | None,
    owner: tuple[BaselineIdentity, bool] | None,
    invocation: StageInvocation,
    summary: StageRunSummary,
) -> None:
    if owner is None:
        adapter.log_parent_stage(parent_run_id, invocation, summary)
        return
    if child_run_id is None:
        raise RuntimeError(f"baseline stage has no MLflow child: {invocation.identifier}")
    adapter.log_baseline_stage(
        child_run_id,
        owner[0],
        invocation,
        summary,
        dependency=owner[1],
    )


def _summary_with_child(
    invocation: StageInvocation,
    child_run_id: str | None,
) -> StageRunSummary | None:
    path = invocation.summary_path
    if not path.is_file():
        return None
    existing = read_stage_summary(path)
    if existing.mlflow_child_run_id == child_run_id:
        return existing
    summary = existing.model_copy(update={"mlflow_child_run_id": child_run_id})
    write_stage_summary(path, summary)
    return summary


def _mark_tracking_or_execution_failure(
    invocation: StageInvocation,
    child_run_id: str | None,
    error: BaseException,
) -> None:
    path = invocation.summary_path
    if not path.is_file():
        return
    try:
        summary = read_stage_summary(path)
    except (OSError, ValueError):
        return
    failed = FailedStageRunSummary.model_validate(
        {
            **summary.model_dump(exclude={"status", "ended_at", "error"}),
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
