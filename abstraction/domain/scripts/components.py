from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.common.capability_names import RequestKind, ViewKind
from abstraction.domain.datasets.split_policy import SplitPolicy
from abstraction.domain.evaluation.metrics import MetricResultTable
from abstraction.domain.graphs.artifacts import GraphArtifact, GraphIndexView
from abstraction.domain.graphs.rules import GraphRuleSet
from abstraction.domain.retrieval.capabilities import MethodCapability
from abstraction.domain.retrieval.methods import (
    AnsweringMethod,
    ContextGatheringMethod,
    GraphRankingMethod,
    TemporalMemoryRankingMethod,
    TextRankingMethod,
)
from abstraction.domain.retrieval.requests import (
    AnswerRequest,
    ContextGatheringRequest,
    GraphRankingRequest,
    RetrievalRequest,
    TemporalMemoryRankingRequest,
    TextRankingRequest,
)
from abstraction.domain.scripts.retrieval_artifacts import TemporalMemorySignals
from abstraction.domain.task_views.eval_views import EvalLabelView
from abstraction.domain.task_views.views import GraphBuildView, TaskView
from abstraction.domain.training.ports import TrainArtifactContract
from abstraction.domain.workflow.manifest import ArtifactManifest, ExperimentRunIntent
from abstraction.domain.workflow.stages import StageGraph


class TaskViewSelector(Protocol):
    def select_request_source_view(
        self,
        task_views: Sequence[TaskView],
        method_capability: MethodCapability,
    ) -> TaskView:
        ...


class EvalViewSelector(Protocol):
    def select_metric_label_view(self, eval_views: Sequence[EvalLabelView]) -> EvalLabelView:
        ...


class GraphRuleSetSelector(Protocol):
    def select_graph_build_inputs(
        self,
        task_views: Sequence[TaskView],
        method_capability: MethodCapability,
    ) -> tuple[GraphBuildView, GraphRuleSet]:
        ...


class RetrievalRequestAssembler(Protocol):
    def as_text_ranking_request(self, projected_request: RetrievalRequest) -> TextRankingRequest:
        ...

    def as_trained_text_ranking_request(
        self,
        projected_request: RetrievalRequest,
        train_artifact: TrainArtifactContract,
    ) -> TextRankingRequest:
        ...

    def as_graph_ranking_request(
        self,
        projected_request: RetrievalRequest,
        graph_artifact: GraphArtifact,
        graph_index: GraphIndexView,
    ) -> GraphRankingRequest:
        ...

    def as_temporal_memory_ranking_request(
        self,
        projected_request: RetrievalRequest,
        temporal_signals: TemporalMemorySignals,
    ) -> TemporalMemoryRankingRequest:
        ...

    def as_context_gathering_request(
        self,
        projected_request: RetrievalRequest,
        graph_artifact: GraphArtifact | None,
    ) -> ContextGatheringRequest:
        ...

    def as_answer_request(self, projected_request: RetrievalRequest) -> AnswerRequest:
        ...


class MethodRuntimeResolver(Protocol):
    def get_text_ranking_method(
        self,
        method_capability: MethodCapability,
        train_artifact: TrainArtifactContract | None,
    ) -> TextRankingMethod:
        ...

    def get_graph_ranking_method(self, method_capability: MethodCapability) -> GraphRankingMethod:
        ...

    def get_temporal_memory_method(self, method_capability: MethodCapability) -> TemporalMemoryRankingMethod:
        ...

    def get_context_gathering_method(self, method_capability: MethodCapability) -> ContextGatheringMethod:
        ...

    def get_answering_method(self, method_capability: MethodCapability) -> AnsweringMethod:
        ...


class ArtifactManifestBuilder(Protocol):
    def build_manifest(
        self,
        intent: ExperimentRunIntent,
        split_policy: SplitPolicy,
        stage_graph: StageGraph,
        metric_result: MetricResultTable,
    ) -> ArtifactManifest:
        ...


class CapabilityTaskViewSelector:  # implement TaskViewSelector
    def select_request_source_view(
        self,
        task_views: Sequence[TaskView],
        method_capability: MethodCapability,
    ) -> TaskView:
        required_view_kind_by_request = {
            RequestKind.TEXT_RANKING: (ViewKind.EVIDENCE_RANKING, ViewKind.CONTEXT_GATHERING),
            RequestKind.GRAPH_RANKING: (ViewKind.GRAPH_BUILD,),
            RequestKind.TEMPORAL_MEMORY_RANKING: (ViewKind.CONTEXT_GATHERING,),
            RequestKind.CONTEXT_GATHERING: (ViewKind.CONTEXT_GATHERING, ViewKind.GRAPH_BUILD),
            RequestKind.ANSWER: (ViewKind.ANSWER_EVALUATION,),
        }
        accepted_view_kinds = required_view_kind_by_request[method_capability.consumed_request_kind]
        for task_view in task_views:
            if task_view.view_kind in accepted_view_kinds:
                return task_view
        raise NotImplementedError
class MetricEvalViewSelector:  # implement EvalViewSelector
    def select_metric_label_view(self, eval_views: Sequence[EvalLabelView]) -> EvalLabelView:
        return eval_views[0]


class CapabilityGraphRuleSetSelector:  # implement GraphRuleSetSelector
    def select_graph_build_inputs(
        self,
        task_views: Sequence[TaskView],
        method_capability: MethodCapability,
    ) -> tuple[GraphBuildView, GraphRuleSet]:
        raise NotImplementedError
class CapabilityRetrievalRequestAssembler:  # implement RetrievalRequestAssembler
    def as_text_ranking_request(self, projected_request: RetrievalRequest) -> TextRankingRequest:
        if not isinstance(projected_request, TextRankingRequest):
            raise TypeError("Expected TextRankingRequest.")
        return projected_request

    def as_trained_text_ranking_request(
        self,
        projected_request: RetrievalRequest,
        train_artifact: TrainArtifactContract,
    ) -> TextRankingRequest:
        text_request = self.as_text_ranking_request(projected_request)
        self._assert_train_artifact_matches_request(train_artifact, text_request)
        return text_request

    def as_graph_ranking_request(
        self,
        projected_request: RetrievalRequest,
        graph_artifact: GraphArtifact,
        graph_index: GraphIndexView,
    ) -> GraphRankingRequest:
        if not isinstance(projected_request, GraphRankingRequest):
            raise TypeError("Expected GraphRankingRequest.")
        return GraphRankingRequest(
            request_id=projected_request.request_id,
            request_kind=RequestKind.GRAPH_RANKING,
            task_id=projected_request.task_id,
            query_ref=projected_request.query_ref,
            candidate_item_ids=graph_index.candidate_item_ids,
            graph_ref=graph_index.graph_ref,
            seed_scores_by_item=projected_request.seed_scores_by_item,
        )

    def as_temporal_memory_ranking_request(
        self,
        projected_request: RetrievalRequest,
        temporal_signals: TemporalMemorySignals,
    ) -> TemporalMemoryRankingRequest:
        if not isinstance(projected_request, TemporalMemoryRankingRequest):
            raise TypeError("Expected TemporalMemoryRankingRequest.")
        return TemporalMemoryRankingRequest(
            request_id=projected_request.request_id,
            request_kind=RequestKind.TEMPORAL_MEMORY_RANKING,
            task_id=projected_request.task_id,
            query_text=projected_request.query_text,
            memory_item_ids=projected_request.memory_item_ids,
            memory_text_by_item=projected_request.memory_text_by_item,
            recency_signal_by_item=temporal_signals.recency_signal_by_item,
            importance_signal_by_item=temporal_signals.importance_signal_by_item,
        )

    def as_context_gathering_request(
        self,
        projected_request: RetrievalRequest,
        graph_artifact: GraphArtifact | None,
    ) -> ContextGatheringRequest:
        if not isinstance(projected_request, ContextGatheringRequest):
            raise TypeError("Expected ContextGatheringRequest.")
        return ContextGatheringRequest(
            request_id=projected_request.request_id,
            request_kind=RequestKind.CONTEXT_GATHERING,
            task_id=projected_request.task_id,
            question_text=projected_request.question_text,
            text_store_ref=projected_request.text_store_ref,
            graph_or_session_context_ref=self._graph_context_ref(graph_artifact),
            candidate_context_item_ids=projected_request.candidate_context_item_ids,
            candidate_text_by_item=projected_request.candidate_text_by_item,
        )

    def as_answer_request(self, projected_request: RetrievalRequest) -> AnswerRequest:
        if not isinstance(projected_request, AnswerRequest):
            raise TypeError("Expected AnswerRequest.")
        return projected_request

    def _assert_train_artifact_matches_request(
        self,
        train_artifact: TrainArtifactContract,
        text_request: TextRankingRequest,
    ) -> None:
        raise NotImplementedError

    def _graph_context_ref(self, graph_artifact: GraphArtifact | None) -> str | None:
        return None if graph_artifact is None else graph_artifact.artifact_id.value

class RegistryMethodRuntimeResolver:  # implement MethodRuntimeResolver
    def get_text_ranking_method(
        self,
        method_capability: MethodCapability,
        train_artifact: TrainArtifactContract | None,
    ) -> TextRankingMethod:
        if train_artifact is not None:
            return self._load_trained_text_ranking_method(method_capability, train_artifact)
        return self._load_text_ranking_method(method_capability)

    def get_graph_ranking_method(self, method_capability: MethodCapability) -> GraphRankingMethod:
        if method_capability.requires_training_artifact:
            return self._load_trained_graph_ranking_method(method_capability)
        return self._load_graph_ranking_method(method_capability)

    def get_temporal_memory_method(self, method_capability: MethodCapability) -> TemporalMemoryRankingMethod:
        return self._load_temporal_memory_method(method_capability)

    def get_context_gathering_method(self, method_capability: MethodCapability) -> ContextGatheringMethod:
        return self._load_context_gathering_method(method_capability)

    def get_answering_method(self, method_capability: MethodCapability) -> AnsweringMethod:
        return self._load_answering_method(method_capability)

    def _load_text_ranking_method(self, method_capability: MethodCapability) -> TextRankingMethod:
        raise NotImplementedError
    def _load_trained_text_ranking_method(
        self,
        method_capability: MethodCapability,
        train_artifact: TrainArtifactContract,
    ) -> TextRankingMethod:
        raise NotImplementedError
    def _load_graph_ranking_method(self, method_capability: MethodCapability) -> GraphRankingMethod:
        raise NotImplementedError
    def _load_trained_graph_ranking_method(self, method_capability: MethodCapability) -> GraphRankingMethod:
        raise NotImplementedError
    def _load_temporal_memory_method(self, method_capability: MethodCapability) -> TemporalMemoryRankingMethod:
        raise NotImplementedError
    def _load_context_gathering_method(self, method_capability: MethodCapability) -> ContextGatheringMethod:
        raise NotImplementedError
    def _load_answering_method(self, method_capability: MethodCapability) -> AnsweringMethod:
        raise NotImplementedError
class ManifestFromStageGraphBuilder:  # implement ArtifactManifestBuilder
    def build_manifest(
        self,
        intent: ExperimentRunIntent,
        split_policy: SplitPolicy,
        stage_graph: StageGraph,
        metric_result: MetricResultTable,
    ) -> ArtifactManifest:
        return ArtifactManifest(
            run_intent=intent,
            split_policy=split_policy,
            stage_artifacts=[],
            task_set_digest_by_stage={
                stage.stage_id: stage.boundary.config_fields.get("task_set_digest", "")
                for stage in stage_graph.stages
            },
        )
