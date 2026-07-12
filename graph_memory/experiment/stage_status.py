from __future__ import annotations

from pathlib import Path
from typing import Literal

from graph_memory.experiment.config import ArtifactBinding, ClosedModel, PublicStageName
from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.state import (
    SuccessfulStageRunSummary,
    read_stage_summary,
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


def inspect_invocation_status(invocation: StageInvocation) -> StatusRow:
    primary = invocation.primary_output
    output_validity = tuple(_artifact_is_valid(output) for output in invocation.outputs)
    if not any(output_validity):
        return _row(
            invocation,
            state="missing",
            path=primary.path,
            reason="outputs missing",
        )
    if not all(output_validity):
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason="partial or wrong-kind outputs",
        )
    if not invocation.summary_path.is_file():
        return _row(
            invocation,
            state="stale",
            path=primary.path,
            reason="stage summary missing",
        )
    try:
        summary = read_stage_summary(invocation.summary_path)
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
    return (
        _row(invocation, state="complete", path=primary.path, reason=None)
        if mismatch is None
        else _row(invocation, state="stale", path=primary.path, reason=mismatch)
    )


def _summary_mismatch(
    invocation: StageInvocation,
    summary: SuccessfulStageRunSummary,
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
    if summary.effective_config != invocation.config.model_dump(
        mode="json",
        by_alias=True,
    ):
        return "effective config mismatch"
    if summary.inputs != invocation.inputs:
        return "input bindings mismatch"
    if summary.outputs != invocation.outputs:
        return "output bindings mismatch"
    return None


def _artifact_is_valid(artifact: ArtifactBinding) -> bool:
    return artifact.path.is_file() if artifact.kind == "file" else artifact.path.is_dir()


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


__all__ = ["ArtifactState", "StatusRow", "inspect_invocation_status"]
