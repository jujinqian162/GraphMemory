from __future__ import annotations

from typing import Protocol

from abstraction.domain.retrieval.capabilities import MethodRegistry
from abstraction.domain.scripts.commands import WorkflowCommandPlan
from abstraction.domain.workflow.components import WorkflowCommandPlanner
from abstraction.domain.workflow.manifest import ExperimentRunIntent
from abstraction.domain.workflow.stages import StageGraph


class CapabilityPlanner(Protocol):
    def plan_experiment(self, intent: ExperimentRunIntent) -> WorkflowCommandPlan:
        ...


class StageGraphFactory(Protocol):
    def create_stage_graph(self, intent: ExperimentRunIntent) -> StageGraph:
        ...


class CrossDatasetCapabilityPlanner:  # implement CapabilityPlanner
    def __init__(
        self,
        method_registry: MethodRegistry,
        stage_graph_factory: StageGraphFactory,
        workflow_command_planner: WorkflowCommandPlanner,
    ) -> None:
        self.method_registry = method_registry
        self.stage_graph_factory = stage_graph_factory
        self.workflow_command_planner = workflow_command_planner

    def plan_experiment(self, intent: ExperimentRunIntent) -> WorkflowCommandPlan:
        method_capability = self.method_registry.get_method_capability(intent.method_id)
        stage_graph = self.stage_graph_factory.create_stage_graph(intent)
        return self.workflow_command_planner.plan_script_commands(
            intent=intent,
            stage_graph=stage_graph,
            method_capability=method_capability,
        )


class CapabilityStageGraphFactory:  # implement StageGraphFactory
    def create_stage_graph(self, intent: ExperimentRunIntent) -> StageGraph:
        raise NotImplementedError