from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scripts.workflow.status import inspect_experiment_status
from scripts.workflow.types import ArtifactState, StageCommand, StageId


@dataclass(frozen=True)
class WorkflowStatusKey:
    """Stable key joining a planned command to a live status row."""

    stage: StageId
    method: str | None = None
    split: str | None = None
    variant: str | None = None

@dataclass(frozen=True)
class ResumeDecision:
    """Cache-aware command selection produced from live status evidence."""

    commands: tuple[StageCommand, ...]
    skipped: tuple[StageCommand, ...]
    first_pending: StageCommand | None


StatusRow = Mapping[str, str]


def command_status_key(command: StageCommand) -> WorkflowStatusKey:
    return WorkflowStatusKey(
        stage=command.stage,
        method=command.method,
        split=command.split,
        variant=command.variant,
    )


def row_status_key(row: StatusRow) -> WorkflowStatusKey:
    return WorkflowStatusKey(
        stage=StageId(row["stage"]),
        method=row.get("method"),
        split=row.get("split"),
        variant=row.get("variant"),
    )


def prune_completed_prefix(
    commands: Sequence[StageCommand],
    status_rows: Sequence[StatusRow],
) -> ResumeDecision:
    status_by_key = {row_status_key(row): row["state"] for row in status_rows}
    command_tuple = tuple(commands)
    skipped: list[StageCommand] = []
    for index, command in enumerate(command_tuple):
        state = status_by_key.get(command_status_key(command), ArtifactState.MISSING.value)
        if state not in {ArtifactState.COMPLETE.value, ArtifactState.ALIAS.value}:
            return ResumeDecision(
                commands=command_tuple[index:],
                skipped=tuple(skipped),
                first_pending=command,
            )
        skipped.append(command)
    return ResumeDecision(commands=(), skipped=tuple(skipped), first_pending=None)


def prune_manifest_completed_prefix(
    manifest: dict[str, object],
    commands: Sequence[StageCommand],
) -> ResumeDecision:
    return prune_completed_prefix(commands, inspect_experiment_status(manifest))
