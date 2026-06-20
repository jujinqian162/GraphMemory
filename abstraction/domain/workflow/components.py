from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.common.capability_names import StageKind
from abstraction.domain.retrieval.capabilities import MethodCapability
from abstraction.domain.scripts.commands import ScriptArgument, ScriptCommand, ScriptCommandResult, WorkflowCommandPlan
from abstraction.domain.workflow.manifest import ExperimentRunIntent
from abstraction.domain.workflow.stages import StageGraph, StagePlan


class StageCommandSelector(Protocol):
    def select_script_name(self, stage: StagePlan, method_capability: MethodCapability) -> str:
        ...

    def select_script_arguments(
        self,
        intent: ExperimentRunIntent,
        stage: StagePlan,
        method_capability: MethodCapability,
    ) -> Sequence[ScriptArgument]:
        ...


class WorkflowCommandPlanner(Protocol):
    def plan_script_commands(
        self,
        intent: ExperimentRunIntent,
        stage_graph: StageGraph,
        method_capability: MethodCapability,
    ) -> WorkflowCommandPlan:
        ...


class WorkflowCommandExecutor(Protocol):
    def execute_command_plan(self, plan: WorkflowCommandPlan) -> Sequence[ScriptCommandResult]:
        ...


class StageKindCommandSelector:  # implement StageCommandSelector
    def select_script_name(self, stage: StagePlan, method_capability: MethodCapability) -> str:
        script_name_by_stage_kind = {
            StageKind.PREPARE_DATASET: "prepare_dataset",
            StageKind.BUILD_TASK_VIEW: "prepare_dataset",
            StageKind.PROJECT_REQUEST: "project_request",
            StageKind.BUILD_GRAPH: "build_graph",
            StageKind.TRAIN_METHOD: "train_method",
            StageKind.TUNE_METHOD: "tune_method",
            StageKind.RUN_RETRIEVAL: "retrieve",
            StageKind.PROJECT_EVALUATION: "evaluate",
            StageKind.RUN_METRICS: "evaluate",
        }
        return script_name_by_stage_kind[stage.stage_kind]

    def select_script_arguments(
        self,
        intent: ExperimentRunIntent,
        stage: StagePlan,
        method_capability: MethodCapability,
    ) -> Sequence[ScriptArgument]:
        common_arguments = [
            ScriptArgument("dataset_id", intent.dataset_id.value),
            ScriptArgument("method_id", intent.method_id.value),
            ScriptArgument("metric_suite_id", intent.metric_suite_id.value),
            ScriptArgument("stage_id", stage.stage_id.value),
        ]
        if stage.stage_kind == StageKind.PREPARE_DATASET:
            return [*common_arguments, ScriptArgument("mode", "describe")]
        if stage.stage_kind == StageKind.BUILD_TASK_VIEW:
            return [*common_arguments, ScriptArgument("mode", "build_views")]
        if stage.stage_kind == StageKind.PROJECT_REQUEST:
            return [*common_arguments, ScriptArgument("projection_mode", "method_capability")]
        if stage.stage_kind == StageKind.BUILD_GRAPH:
            return [*common_arguments, ScriptArgument("graph_mode", "build_and_index")]
        if stage.stage_kind == StageKind.TRAIN_METHOD:
            train_mode = "train_if_required" if method_capability.requires_training_artifact else "skip"
            return [*common_arguments, ScriptArgument("train_mode", train_mode)]
        if stage.stage_kind == StageKind.TUNE_METHOD:
            return [*common_arguments, ScriptArgument("tune_mode", "capability_selected")]
        if stage.stage_kind == StageKind.RUN_RETRIEVAL:
            return [*common_arguments, ScriptArgument("retrieval_mode", "rank")]
        if stage.stage_kind == StageKind.PROJECT_EVALUATION:
            return [*common_arguments, ScriptArgument("evaluation_mode", "projection_only")]
        if stage.stage_kind == StageKind.RUN_METRICS:
            return [*common_arguments, ScriptArgument("evaluation_mode", "prediction_to_metric")]
        raise NotImplementedError
class StageGraphCommandPlanner:  # implement WorkflowCommandPlanner
    def __init__(self, stage_command_selector: StageCommandSelector) -> None:
        self.stage_command_selector = stage_command_selector

    def plan_script_commands(
        self,
        intent: ExperimentRunIntent,
        stage_graph: StageGraph,
        method_capability: MethodCapability,
    ) -> WorkflowCommandPlan:
        commands = [
            ScriptCommand(
                stage_id=stage.stage_id,
                stage_kind=stage.stage_kind,
                script_name=self.stage_command_selector.select_script_name(stage, method_capability),
                arguments=self.stage_command_selector.select_script_arguments(intent, stage, method_capability),
                reads_artifacts=stage.boundary.read_artifacts,
                writes_artifacts=stage.boundary.write_artifacts,
            )
            for stage in stage_graph.stages
        ]
        return WorkflowCommandPlan(
            commands=commands,
            command_dependencies={stage.stage_id: stage.depends_on for stage in stage_graph.stages},
        )


class SequentialWorkflowCommandExecutor:  # implement WorkflowCommandExecutor
    def execute_command_plan(self, plan: WorkflowCommandPlan) -> Sequence[ScriptCommandResult]:
        raise NotImplementedError