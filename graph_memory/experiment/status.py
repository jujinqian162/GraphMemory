from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from graph_memory.experiment.config import ArtifactRef, ClosedModel, PublicStageName
from graph_memory.experiment.planning import StageAlias, StageInvocation, WorkflowPlan
from graph_memory.experiment.state import (
    StageRunSummary,
    read_stage_summary,
    summary_path_for,
)
from graph_memory.registry.retrieval import RetrievalMethodId

ArtifactState = Literal["missing", "complete", "stale", "alias"]


class StatusRow(ClosedModel):
    identifier: str
    stage: PublicStageName
    state: ArtifactState
    path: Path
    method: RetrievalMethodId | None
    split: str | None
    variant: str | None
    reason: str | None


class StatusCommandConfig(ClosedModel):
    name: str
    job: str | None = None


def status_named_run(command: StatusCommandConfig) -> str:
    from graph_memory.experiment.service import load_existing_experiment

    existing = load_existing_experiment(command.name, job=command.job)
    return format_status(inspect_plan_status(existing.plan))


def inspect_plan_status(plan: WorkflowPlan) -> tuple[StatusRow, ...]:
    invocation_rows = tuple(
        inspect_invocation_status(item) for item in plan.invocations
    )
    return (*invocation_rows, *(_alias_status(alias) for alias in plan.aliases))


def inspect_invocation_status(invocation: StageInvocation) -> StatusRow:
    primary = invocation.primary_output
    output_validity = tuple(_artifact_is_valid(output) for output in invocation.outputs)
    if not any(output_validity):
        return _row(
            invocation, state="missing", path=primary.path, reason="outputs missing"
        )
    if not all(output_validity):
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason="partial or wrong-kind outputs",
        )

    summary_path = summary_path_for(invocation)
    if not summary_path.is_file():
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason="stage summary missing",
        )
    try:
        summary = read_stage_summary(summary_path)
    except (OSError, ValueError) as error:
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason=f"invalid stage summary: {error}",
        )
    if summary.status != "success":
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason=f"summary status is {summary.status}",
        )
    mismatch = _summary_mismatch(invocation, summary)
    if mismatch is not None:
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason=mismatch,
        )
    return _row(invocation, state="complete", path=primary.path, reason=None)


def format_status(rows: Sequence[StatusRow]) -> str:
    lines: list[str] = []
    for row in rows:
        qualifiers = [row.stage]
        if row.method is not None:
            qualifiers.append(row.method.value)
        if row.split is not None:
            qualifiers.append(row.split)
        if row.variant is not None:
            qualifiers.append(f"variant={row.variant}")
        qualifiers.append(row.state)
        if row.reason:
            qualifiers.append(f"({row.reason})")
        lines.append(" ".join(qualifiers))
    return "\n".join(lines)


def _summary_mismatch(
    invocation: StageInvocation,
    summary: StageRunSummary,
) -> str | None:
    identity = (
        summary.identifier,
        summary.stage,
        summary.script.resolve(),
        summary.method,
        summary.split,
        summary.variant,
    )
    expected_identity = (
        invocation.identifier,
        invocation.stage,
        invocation.script.resolve(),
        invocation.method,
        invocation.split,
        invocation.variant,
    )
    if identity != expected_identity:
        return "summary identity mismatch"
    expected_config = invocation.config.model_dump(mode="json", by_alias=True)
    if summary.effective_config != expected_config:
        return "effective config mismatch"
    if summary.inputs != invocation.inputs:
        return "input bindings mismatch"
    if summary.outputs != invocation.outputs:
        return "output bindings mismatch"
    if summary.ended_at is None or summary.error is not None:
        return "successful summary is incomplete"
    return None


def _artifact_is_valid(artifact: ArtifactRef) -> bool:
    if artifact.kind == "file":
        return artifact.path.is_file()
    return artifact.path.is_dir()


def _row(
    invocation: StageInvocation,
    *,
    state: ArtifactState,
    path: Path,
    reason: str | None,
) -> StatusRow:
    return StatusRow(
        identifier=invocation.identifier,
        stage=invocation.stage,
        state=state,
        path=path,
        method=invocation.method,
        split=invocation.split,
        variant=invocation.variant,
        reason=reason,
    )


def _alias_status(alias: StageAlias) -> StatusRow:
    source = alias.artifact.alias_of
    source_state = inspect_invocation_status(alias.source_invocation).state
    state: ArtifactState = "alias" if source_state == "complete" else source_state
    return StatusRow(
        identifier=alias.identifier,
        stage=alias.stage,
        state=state,
        path=alias.artifact.path,
        method=alias.method,
        split=None,
        variant=alias.variant,
        reason=(
            f"reuses {source}"
            if state == "alias"
            else f"alias source is {source_state}: {source}"
        ),
    )


__all__ = [
    "ArtifactState",
    "StatusRow",
    "StatusCommandConfig",
    "format_status",
    "inspect_invocation_status",
    "inspect_plan_status",
    "status_named_run",
]
