from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graph_memory.experiment.invocation import StageInvocation
from graph_memory.experiment.planning import WorkflowPlan
from graph_memory.experiment.status import StatusRow, inspect_plan_status


@dataclass(frozen=True)
class ResumeDecision:
    invocations: tuple[StageInvocation, ...]
    skipped: tuple[StageInvocation, ...]
    first_pending: StageInvocation | None


def prune_completed_prefix(
    invocations: Sequence[StageInvocation],
    rows: Sequence[StatusRow],
    *,
    cache_enabled: bool,
) -> ResumeDecision:
    ordered = tuple(invocations)
    if not cache_enabled:
        return ResumeDecision(
            invocations=ordered,
            skipped=(),
            first_pending=ordered[0] if ordered else None,
        )
    states = {row.identifier: row.state for row in rows}
    skipped: list[StageInvocation] = []
    for index, invocation in enumerate(ordered):
        if states.get(invocation.identifier, "missing") not in {"complete", "alias"}:
            return ResumeDecision(
                invocations=ordered[index:],
                skipped=tuple(skipped),
                first_pending=invocation,
            )
        skipped.append(invocation)
    return ResumeDecision(invocations=(), skipped=tuple(skipped), first_pending=None)


def resume_plan(plan: WorkflowPlan, *, cache_enabled: bool) -> ResumeDecision:
    return prune_completed_prefix(
        plan.invocations,
        inspect_plan_status(plan),
        cache_enabled=cache_enabled,
    )


__all__ = ["ResumeDecision", "prune_completed_prefix", "resume_plan"]
