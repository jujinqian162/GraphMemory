from __future__ import annotations

from collections.abc import Sequence

from graph_memory.experiment.config import ClosedModel
from graph_memory.experiment.planning import StageAlias, WorkflowPlan
from graph_memory.experiment.stage_status import (
    ArtifactState,
    StatusRow,
    inspect_invocation_status,
)


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
    "StatusCommandConfig",
    "StatusRow",
    "format_status",
    "inspect_invocation_status",
    "inspect_plan_status",
    "status_named_run",
]
