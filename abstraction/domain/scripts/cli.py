from __future__ import annotations

from typing import Mapping, Protocol

from abstraction.domain.common.capability_names import RequestKind
from abstraction.domain.scripts.cli_arguments import ScriptCliArguments
from abstraction.domain.scripts.commands import ScriptCommandResult
from abstraction.domain.scripts.composition import (
    RetrieveScriptContext,
    ScriptCompositionRoot,
    ScriptLocalCompositionRoot,
)
from abstraction.domain.scripts.steps import RetrievalResult


def build_script_composition_root() -> ScriptCompositionRoot:
    return ScriptLocalCompositionRoot()


class ScriptCliEntrypoint(Protocol):
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        ...


class RunPrepareDatasetScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_prepare_dataset_context(args)
        mode = args.flags.get("mode", "build_views")
        if mode == "describe":
            context.dataset_preparation.describe_dataset(args.intent)
            return self._result(args, selected_branch="describe_dataset_only")
        if mode == "build_views":
            prepared_dataset = context.dataset_preparation.prepare_dataset(args.intent)
            context.artifact_writer.write_prepared_dataset(prepared_dataset, args.command.writes_artifacts)
            return self._result(args, selected_branch="load_raw_build_task_and_eval_views")
        if mode == "validate_views":
            prepared_dataset = context.dataset_preparation.prepare_dataset(args.intent)
            context.artifact_writer.write_prepared_dataset(prepared_dataset, args.command.writes_artifacts)
            return self._result(args, selected_branch="build_views_then_validate_boundaries")
        raise NotImplementedError
    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class RunProjectRequestScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_project_request_context(args)
        prepared_dataset = context.artifact_reader.load_prepared_dataset(args.command.reads_artifacts)
        projection_mode = args.flags.get("projection_mode", "method_capability")
        if projection_mode == "method_capability":
            request_result = context.request_projection.project_request(args.intent, prepared_dataset.task_views)
            context.artifact_writer.write_projected_request(request_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="select_source_view_from_method_capability")
        if projection_mode == "explicit_view":
            request_result = context.request_projection.project_request(args.intent, prepared_dataset.task_views)
            context.artifact_writer.write_projected_request(request_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="use_cli_selected_view_kind")
        raise NotImplementedError
    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class RunBuildGraphScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_graph_context(args)
        prepared_dataset = context.artifact_reader.load_prepared_dataset(args.command.reads_artifacts)
        method_capability = context.method_registry.get_method_capability(args.intent.method_id)
        graph_mode = args.flags.get("graph_mode", "build_and_index")
        if graph_mode == "skip_if_not_required":
            graph_result = context.graph_build.build_graph_for_method(prepared_dataset.task_views, method_capability)
            context.artifact_writer.write_graph_result(graph_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="only_build_when_method_requires_graph")
        if graph_mode == "build_and_index":
            graph_result = context.graph_build.build_graph_for_method(prepared_dataset.task_views, method_capability)
            context.artifact_writer.write_graph_result(graph_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="build_graph_then_index")
        if graph_mode == "index_existing":
            graph_result = context.artifact_reader.load_graph_result(args.command.reads_artifacts)
            context.artifact_writer.write_graph_result(graph_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="load_existing_graph_then_index")
        raise NotImplementedError
    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class RunTrainMethodScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_train_context(args)
        prepared_dataset = context.artifact_reader.load_prepared_dataset(args.command.reads_artifacts)
        method_capability = context.method_registry.get_method_capability(args.intent.method_id)
        graph_result = context.artifact_reader.load_graph_result(args.command.reads_artifacts)
        train_mode = args.flags.get("train_mode", "train_if_required")
        if train_mode == "skip":
            return self._result(args, selected_branch="explicit_cli_skip")
        if train_mode == "train_if_required":
            train_result = context.training.train_method_if_required(prepared_dataset.task_views, graph_result, method_capability)
            context.artifact_writer.write_training_result(train_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="train_only_when_capability_requires_artifact")
        if train_mode == "force_train":
            train_result = context.training.train_method_if_required(prepared_dataset.task_views, graph_result, method_capability)
            context.artifact_writer.write_training_result(train_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="force_training_adapter_path")
        raise NotImplementedError
    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class RunTuneMethodScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_tune_context(args)
        method_capability = context.method_registry.get_method_capability(args.intent.method_id)
        tune_mode = args.flags.get("tune_mode", "capability_selected")
        if tune_mode == "skip":
            return self._result(args, selected_branch="explicit_cli_skip")
        if tune_mode == "fixed_config":
            return self._result(args, selected_branch="write_fixed_selected_config")
        if tune_mode == "capability_selected" and method_capability.requires_tuning_artifact:
            return self._result(args, selected_branch="run_method_declared_tuning_adapter")
        if tune_mode == "capability_selected":
            return self._result(args, selected_branch="method_does_not_require_tuning")
        raise NotImplementedError
    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class RunRetrieveStageScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_retrieve_context(args)
        method_capability = context.method_registry.get_method_capability(args.intent.method_id)
        request_result = context.artifact_reader.load_projected_request(args.command.reads_artifacts)
        retrieval_mode = args.flags.get("retrieval_mode", "rank")
        if retrieval_mode == "dry_run_request":
            return self._result(args, selected_branch="project_request_without_method_execution")
        if retrieval_mode == "rank" and method_capability.consumed_request_kind == RequestKind.TEXT_RANKING:
            if method_capability.requires_training_artifact:
                training_result = context.artifact_reader.load_required_training_result(args.command.reads_artifacts)
                retrieval_result = context.retrieval.run_trained_text_ranking_stage(
                    method_capability,
                    request_result,
                    training_result,
                )
                return self._write_retrieval_result(args, context, retrieval_result, "trained_text_rank_task")
            retrieval_result = context.retrieval.run_text_ranking_stage(method_capability, request_result)
            return self._write_retrieval_result(args, context, retrieval_result, "text_rank_task")
        if retrieval_mode == "rank" and method_capability.consumed_request_kind == RequestKind.GRAPH_RANKING:
            graph_result = context.artifact_reader.load_graph_result(args.command.reads_artifacts)
            if graph_result is None:
                raise ValueError("Graph ranking requires a graph artifact.")
            retrieval_result = context.retrieval.run_graph_ranking_stage(
                method_capability,
                request_result,
                graph_result,
            )
            return self._write_retrieval_result(args, context, retrieval_result, "graph_rank_task")
        if retrieval_mode == "rank" and method_capability.consumed_request_kind == RequestKind.TEMPORAL_MEMORY_RANKING:
            temporal_signals = context.artifact_reader.load_temporal_memory_signals(args.command.reads_artifacts)
            retrieval_result = context.retrieval.run_temporal_memory_ranking_stage(
                method_capability,
                request_result,
                temporal_signals,
            )
            return self._write_retrieval_result(args, context, retrieval_result, "temporal_memory_rank_task")
        if retrieval_mode == "rank" and method_capability.consumed_request_kind == RequestKind.CONTEXT_GATHERING:
            graph_result = (
                context.artifact_reader.load_graph_result(args.command.reads_artifacts)
                if method_capability.requires_graph_artifact
                else None
            )
            retrieval_result = context.retrieval.run_context_gathering_stage(
                method_capability,
                request_result,
                graph_result,
            )
            return self._write_retrieval_result(args, context, retrieval_result, "context_gathering_task")
        if retrieval_mode == "rank" and method_capability.consumed_request_kind == RequestKind.ANSWER:
            retrieval_result = context.retrieval.run_answering_stage(method_capability, request_result)
            return self._write_retrieval_result(args, context, retrieval_result, "answer_task")
        raise NotImplementedError
    def _write_retrieval_result(
        self,
        args: ScriptCliArguments,
        context: RetrieveScriptContext,
        retrieval_result: RetrievalResult,
        selected_branch: str,
    ) -> ScriptCommandResult:
        context.artifact_writer.write_retrieval_result(retrieval_result, args.command.writes_artifacts)
        return self._result(args, selected_branch=selected_branch)

    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class RunEvaluateStageScript:  # implement ScriptCliEntrypoint
    def run(self, args: ScriptCliArguments) -> ScriptCommandResult:
        context = build_script_composition_root().build_evaluate_context(args)
        prepared_dataset = context.artifact_reader.load_prepared_dataset(args.command.reads_artifacts)
        method_capability = context.method_registry.get_method_capability(args.intent.method_id)
        context.boundary_review.review_prepared_step_boundaries(prepared_dataset, method_capability, args.stage_graph)
        graph_result = context.artifact_reader.load_graph_result(args.command.reads_artifacts)
        retrieval_prediction, _ = context.artifact_reader.load_retrieval_result(args.command.reads_artifacts)
        evaluation_mode = args.flags.get("evaluation_mode", "prediction_to_metric")
        if evaluation_mode == "projection_only":
            evaluation_result = context.evaluation.evaluate_prediction(
                args.intent,
                method_capability,
                retrieval_prediction,
                prepared_dataset.eval_views,
                graph_result,
            )
            context.artifact_writer.write_evaluation_result(evaluation_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="project_prediction_to_eval_units_only")
        if evaluation_mode == "prediction_to_metric":
            evaluation_result = context.evaluation.evaluate_prediction(
                args.intent,
                method_capability,
                retrieval_prediction,
                prepared_dataset.eval_views,
                graph_result,
            )
            context.artifact_writer.write_evaluation_result(evaluation_result, args.command.writes_artifacts)
            return self._result(args, selected_branch="project_eval_units_then_metric_suite")
        if evaluation_mode == "publish_manifest":
            evaluation_result = context.artifact_reader.load_evaluation_result(args.command.reads_artifacts)
            context.artifact_publication.publish_stage_artifacts(artifacts=[])
            return self._result(args, selected_branch="evaluate_then_publish_artifacts")
        raise NotImplementedError
    def _result(self, args: ScriptCliArguments, selected_branch: str) -> ScriptCommandResult:
        return ScriptCommandResult(args.command, args.command.writes_artifacts, selected_branch)


class ScriptCliRouter:
    def __init__(self, entrypoint_by_script_name: Mapping[str, ScriptCliEntrypoint]) -> None:
        self.entrypoint_by_script_name = entrypoint_by_script_name

    def dispatch(self, args: ScriptCliArguments) -> ScriptCommandResult:
        entrypoint = self.entrypoint_by_script_name[args.command.script_name]
        return entrypoint.run(args)
