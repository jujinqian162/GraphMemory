from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

from abstraction.domain.common.artifacts import ArtifactRef
from abstraction.domain.common.identifiers import TaskId
from abstraction.domain.datasets.definitions import (
    AssetManifest,
    DatasetDefinition,
    DatasetRecordSet,
    OfficialSplitMetadata,
)
from abstraction.domain.datasets.ports import DatasetRegistry
from abstraction.domain.datasets.split_policy import SplitPolicy, SplitPolicyResolver
from abstraction.domain.evaluation.metrics import MetricResultTable
from abstraction.domain.evaluation.suites import MetricSuiteRegistry
from abstraction.domain.graphs.artifacts import GraphArtifact, GraphIndexView
from abstraction.domain.graphs.builder import GraphBuilder, GraphIndexBuilder
from abstraction.domain.projections.eval_units import EvaluationUnitBatch
from abstraction.domain.projections.ports import ProjectionAdapter, ProjectionDefinition, ProjectionPlanner, ProjectionRegistry
from abstraction.domain.retrieval.capabilities import MethodCapability, MethodRegistry
from abstraction.domain.retrieval.predictions import RetrievalPrediction
from abstraction.domain.retrieval.requests import RetrievalRequest
from abstraction.domain.scripts.components import (
    EvalViewSelector,
    GraphRuleSetSelector,
    MethodRuntimeResolver,
    RetrievalRequestAssembler,
    TaskViewSelector,
)
from abstraction.domain.scripts.retrieval_artifacts import TemporalMemorySignals
from abstraction.domain.task_views.eval_views import EvalLabelView
from abstraction.domain.task_views.ports import EvalViewCatalog, TaskViewCatalog, TaskViewValidator
from abstraction.domain.task_views.views import TaskView
from abstraction.domain.training.ports import TrainArtifactContract, TrainableMethodAdapter, TrainingViewBuilder
from abstraction.domain.workflow.guards import AssetCoverageGuard, LabelVisibilityGuard, SplitAlignmentGuard
from abstraction.domain.workflow.manifest import ExperimentRunIntent
from abstraction.domain.workflow.scenarios import ScenarioCompatibilityReviewer, ScenarioFlow
from abstraction.domain.workflow.stages import StageGraph


@dataclass(frozen=True)
class PreparedDatasetResult:
    dataset_definition: DatasetDefinition
    official_splits: OfficialSplitMetadata
    split_policy: SplitPolicy
    records: DatasetRecordSet
    assets: AssetManifest
    task_views: Sequence[TaskView]
    eval_views: Sequence[EvalLabelView]


ProjectedRequestResult: TypeAlias = tuple[
    TaskView,
    ProjectionAdapter[TaskView, RetrievalRequest],
    RetrievalRequest,
]
GraphBuildResult: TypeAlias = tuple[GraphArtifact, GraphIndexView]
TrainingResult: TypeAlias = tuple[TrainArtifactContract, Sequence[TaskId]]
RetrievalResult: TypeAlias = tuple[RetrievalPrediction, RetrievalRequest]
EvaluationResult: TypeAlias = tuple[EvaluationUnitBatch, MetricResultTable]


class DatasetPreparationStep:
    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        split_policy_resolver: SplitPolicyResolver,
        task_view_validator: TaskViewValidator,
    ) -> None:
        self.dataset_registry = dataset_registry
        self.split_policy_resolver = split_policy_resolver
        self.task_view_validator = task_view_validator

    def describe_dataset(self, intent: ExperimentRunIntent) -> tuple[DatasetDefinition, OfficialSplitMetadata]:
        adapter = self.dataset_registry.get_dataset(intent.dataset_id)
        return adapter.describe_dataset(), adapter.describe_official_splits(intent.raw_source)

    def prepare_dataset(self, intent: ExperimentRunIntent) -> PreparedDatasetResult:
        adapter = self.dataset_registry.get_dataset(intent.dataset_id)
        dataset_definition = adapter.describe_dataset()
        official_splits = adapter.describe_official_splits(intent.raw_source)
        split_policy = self.split_policy_resolver.resolve_split_policy(
            intent.benchmark_recipe,
            intent.method_id,
        )
        records = adapter.load_raw_records(intent.raw_source, split_policy)
        assets = adapter.describe_assets(intent.raw_source, split_policy)
        task_views = adapter.build_task_views(records, split_policy)
        eval_views = adapter.build_eval_views(records, split_policy)

        for task_view in task_views:
            self.task_view_validator.validate_task_view(task_view)
        for eval_view in eval_views:
            self.task_view_validator.validate_eval_view(eval_view)

        return PreparedDatasetResult(
            dataset_definition=dataset_definition,
            official_splits=official_splits,
            split_policy=split_policy,
            records=records,
            assets=assets,
            task_views=task_views,
            eval_views=eval_views,
        )


class TaskViewLookupStep:
    def __init__(self, task_view_catalog: TaskViewCatalog, eval_view_catalog: EvalViewCatalog) -> None:
        self.task_view_catalog = task_view_catalog
        self.eval_view_catalog = eval_view_catalog

    def load_task_and_label_view(self, intent: ExperimentRunIntent, task_id: TaskId) -> tuple[TaskView, EvalLabelView]:
        available_view_kinds = self.task_view_catalog.list_view_kinds(intent.dataset_id)
        task_view = self.task_view_catalog.get_task_view(intent.dataset_id, task_id, available_view_kinds[0])
        eval_view = self.eval_view_catalog.get_eval_view(intent.dataset_id, task_id)
        return task_view, eval_view


class RequestProjectionStep:
    def __init__(
        self,
        method_registry: MethodRegistry,
        projection_registry: ProjectionRegistry,
        task_view_selector: TaskViewSelector,
    ) -> None:
        self.method_registry = method_registry
        self.projection_registry = projection_registry
        self.task_view_selector = task_view_selector

    def project_request(self, intent: ExperimentRunIntent, task_views: Sequence[TaskView]) -> ProjectedRequestResult:
        method_capability = self.method_registry.get_method_capability(intent.method_id)
        source_view = self.task_view_selector.select_request_source_view(task_views, method_capability)
        request_projection = self.projection_registry.find_task_to_request_projection(
            source_view_kind=source_view.view_kind,
            target_request_kind=method_capability.consumed_request_kind,
            dataset_id=intent.dataset_id,
        )
        request_projection.describe_projection()
        retrieval_request = request_projection.project(source_view)
        return source_view, request_projection, retrieval_request


class GraphBuildStep:
    def __init__(
        self,
        graph_rule_set_selector: GraphRuleSetSelector,
        graph_builder: GraphBuilder,
        graph_index_builder: GraphIndexBuilder,
    ) -> None:
        self.graph_rule_set_selector = graph_rule_set_selector
        self.graph_builder = graph_builder
        self.graph_index_builder = graph_index_builder

    def build_graph_for_method(
        self,
        task_views: Sequence[TaskView],
        method_capability: MethodCapability,
    ) -> GraphBuildResult | None:
        if not method_capability.requires_graph_artifact:
            return None
        graph_view, graph_rule_set = self.graph_rule_set_selector.select_graph_build_inputs(
            task_views,
            method_capability,
        )
        graph_artifact = self.graph_builder.build_graph(graph_view, graph_rule_set)
        graph_index = self.graph_index_builder.build_graph_index(graph_artifact)
        return graph_artifact, graph_index


class TrainingStep:
    def __init__(
        self,
        training_view_builder: TrainingViewBuilder,
        trainable_method_adapter: TrainableMethodAdapter,
    ) -> None:
        self.training_view_builder = training_view_builder
        self.trainable_method_adapter = trainable_method_adapter

    def train_method_if_required(
        self,
        source_views: Sequence[TaskView],
        graph_result: GraphBuildResult | None,
        method_capability: MethodCapability,
    ) -> TrainingResult | None:
        if not method_capability.requires_training_artifact:
            return None
        training_views = [self.training_view_builder.build_training_view(view) for view in source_views]
        optional_graphs = [] if graph_result is None else [graph_result[0]]
        train_artifact = self.trainable_method_adapter.train_method(training_views, optional_graphs)
        return train_artifact, [view.task_id for view in source_views]


class RetrievalStep:
    def __init__(
        self,
        request_assembler: RetrievalRequestAssembler,
        method_runtime_resolver: MethodRuntimeResolver,
    ) -> None:
        self.request_assembler = request_assembler
        self.method_runtime_resolver = method_runtime_resolver

    def run_text_ranking_stage(
        self,
        method_capability: MethodCapability,
        request_result: ProjectedRequestResult,
    ) -> RetrievalResult:
        _, _, retrieval_request = request_result
        request = self.request_assembler.as_text_ranking_request(retrieval_request)
        method = self.method_runtime_resolver.get_text_ranking_method(method_capability, train_artifact=None)
        prediction = method.rank_task(request)
        return prediction, request

    def run_trained_text_ranking_stage(
        self,
        method_capability: MethodCapability,
        request_result: ProjectedRequestResult,
        training_result: TrainingResult,
    ) -> RetrievalResult:
        _, _, retrieval_request = request_result
        train_artifact, _ = training_result
        request = self.request_assembler.as_trained_text_ranking_request(retrieval_request, train_artifact)
        method = self.method_runtime_resolver.get_text_ranking_method(method_capability, train_artifact)
        prediction = method.rank_task(request)
        return prediction, request

    def run_graph_ranking_stage(
        self,
        method_capability: MethodCapability,
        request_result: ProjectedRequestResult,
        graph_result: GraphBuildResult,
    ) -> RetrievalResult:
        _, _, retrieval_request = request_result
        request = self.request_assembler.as_graph_ranking_request(
            retrieval_request,
            graph_result[0],
            graph_result[1],
        )
        method = self.method_runtime_resolver.get_graph_ranking_method(method_capability)
        prediction = method.rank_task(request)
        return prediction, request

    def run_temporal_memory_ranking_stage(
        self,
        method_capability: MethodCapability,
        request_result: ProjectedRequestResult,
        temporal_signals: TemporalMemorySignals,
    ) -> RetrievalResult:
        _, _, retrieval_request = request_result
        request = self.request_assembler.as_temporal_memory_ranking_request(
            retrieval_request,
            temporal_signals,
        )
        method = self.method_runtime_resolver.get_temporal_memory_method(method_capability)
        prediction = method.rank_task(request)
        return prediction, request

    def run_context_gathering_stage(
        self,
        method_capability: MethodCapability,
        request_result: ProjectedRequestResult,
        graph_result: GraphBuildResult | None,
    ) -> RetrievalResult:
        _, _, retrieval_request = request_result
        request = self.request_assembler.as_context_gathering_request(
            retrieval_request,
            None if graph_result is None else graph_result[0],
        )
        method = self.method_runtime_resolver.get_context_gathering_method(method_capability)
        prediction = method.gather_task_context(request)
        return prediction, request

    def run_answering_stage(
        self,
        method_capability: MethodCapability,
        request_result: ProjectedRequestResult,
    ) -> RetrievalResult:
        _, _, retrieval_request = request_result
        request = self.request_assembler.as_answer_request(retrieval_request)
        method = self.method_runtime_resolver.get_answering_method(method_capability)
        prediction = method.answer_task(request)
        return prediction, request


class EvaluationStep:
    def __init__(
        self,
        projection_registry: ProjectionRegistry,
        metric_suite_registry: MetricSuiteRegistry,
        eval_view_selector: EvalViewSelector,
    ) -> None:
        self.projection_registry = projection_registry
        self.metric_suite_registry = metric_suite_registry
        self.eval_view_selector = eval_view_selector

    def evaluate_prediction(
        self,
        intent: ExperimentRunIntent,
        method_capability: MethodCapability,
        prediction: RetrievalPrediction,
        eval_views: Sequence[EvalLabelView],
        graph_result: GraphBuildResult | None,
    ) -> EvaluationResult:
        metric_suite = self.metric_suite_registry.get_metric_suite(intent.metric_suite_id)
        label_view = self.eval_view_selector.select_metric_label_view(eval_views)
        prediction_projection = self.projection_registry.find_prediction_to_eval_projection(
            source_prediction_kind=method_capability.produced_prediction_kind,
            target_eval_unit_kind=metric_suite.describe_metric_suite().eval_unit_kind,
            dataset_id=intent.dataset_id,
        )
        prediction_projection.describe_projection()
        eval_units = prediction_projection.project((prediction, label_view, None if graph_result is None else graph_result[0]))
        metric_result = metric_suite.evaluate_units(eval_units)
        return eval_units, metric_result


class ArtifactPublicationStep:
    def publish_stage_artifacts(self, artifacts: Sequence[ArtifactRef]) -> None:
        raise NotImplementedError
class ScriptBoundaryReviewStep:
    def __init__(
        self,
        label_visibility_guard: LabelVisibilityGuard,
        asset_coverage_guard: AssetCoverageGuard,
        split_alignment_guard: SplitAlignmentGuard,
    ) -> None:
        self.label_visibility_guard = label_visibility_guard
        self.asset_coverage_guard = asset_coverage_guard
        self.split_alignment_guard = split_alignment_guard

    def review_prepared_step_boundaries(
        self,
        prepared_dataset: PreparedDatasetResult,
        method_capability: MethodCapability,
        stage_graph: StageGraph,
    ) -> None:
        self.label_visibility_guard.assert_labels_hidden_from_retrieval(
            prepared_dataset.split_policy.split_role,
            stage_graph,
        )
        self.asset_coverage_guard.assert_asset_coverage(
            method_capability.method_id,
            prepared_dataset.split_policy.selected_task_ids,
            prepared_dataset.assets,
            prepared_dataset.split_policy,
        )
        self.split_alignment_guard.assert_stage_task_sets_aligned(stage_graph, prepared_dataset.split_policy)


class CapabilityReviewStep:
    def __init__(
        self,
        method_registry: MethodRegistry,
        projection_registry: ProjectionRegistry,
        projection_planner: ProjectionPlanner,
        scenario_reviewer: ScenarioCompatibilityReviewer,
    ) -> None:
        self.method_registry = method_registry
        self.projection_registry = projection_registry
        self.projection_planner = projection_planner
        self.scenario_reviewer = scenario_reviewer

    def review_scenario_flow(
        self,
        scenario: ScenarioFlow,
        requested_projection: ProjectionDefinition,
    ) -> Sequence[MethodCapability]:
        projection_chain = self.projection_planner.plan_projection_chain(requested_projection)
        matching_methods = self.method_registry.list_methods_for_request(scenario.describe_required_request())
        for method_capability in matching_methods:
            self.scenario_reviewer.assert_method_matches_scenario(scenario, method_capability)
        scenario.connect_projection(registry=self.projection_registry)
        return matching_methods if projection_chain else []
