from __future__ import annotations

from typing import Sequence

from abstraction.domain.scripts.commands import ScriptCommandResult, WorkflowCommandPlan
from abstraction.domain.workflow.components import WorkflowCommandExecutor
from abstraction.domain.workflow.manifest import ExperimentRunIntent
from abstraction.domain.workflow.planner import CapabilityPlanner


class ExperimentPlanningEntryPoint:
    def __init__(self, capability_planner: CapabilityPlanner) -> None:
        self.capability_planner = capability_planner

    def plan_requested_experiment(self, intent: ExperimentRunIntent) -> WorkflowCommandPlan:
        return self.capability_planner.plan_experiment(intent)


class WorkflowExecutionEntryPoint:
    def __init__(self, command_executor: WorkflowCommandExecutor) -> None:
        self.command_executor = command_executor

    def execute_planned_workflow(self, plan: WorkflowCommandPlan) -> Sequence[ScriptCommandResult]:
        return self.command_executor.execute_command_plan(plan)
