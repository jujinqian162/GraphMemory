from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.common.capability_names import SplitRole
from abstraction.domain.common.identifiers import MethodId, TaskId
from abstraction.domain.datasets.definitions import AssetManifest
from abstraction.domain.datasets.split_policy import SplitPolicy
from abstraction.domain.workflow.stages import StageGraph


class LabelVisibilityGuard(Protocol):
    def assert_labels_hidden_from_retrieval(self, split_role: SplitRole, stage_graph: StageGraph) -> None:
        ...


class AssetCoverageGuard(Protocol):
    def assert_asset_coverage(
        self,
        method_id: MethodId,
        selected_task_ids: Sequence[TaskId],
        asset_manifest: AssetManifest,
        split_policy: SplitPolicy,
    ) -> None:
        ...


class SplitAlignmentGuard(Protocol):
    def assert_stage_task_sets_aligned(self, stage_graph: StageGraph, split_policy: SplitPolicy) -> None:
        ...


class PortLevelLabelVisibilityGuard:  # implement LabelVisibilityGuard
    def assert_labels_hidden_from_retrieval(self, split_role: SplitRole, stage_graph: StageGraph) -> None:
        for stage in stage_graph.stages:
            if stage.boundary.reads_labels and stage.boundary.reads_retrieval_visible_inputs:
                raise AssertionError(f"label-visible stage crosses retrieval boundary: {stage.stage_id.value}")


class ManifestAssetCoverageGuard:  # implement AssetCoverageGuard
    def assert_asset_coverage(
        self,
        method_id: MethodId,
        selected_task_ids: Sequence[TaskId],
        asset_manifest: AssetManifest,
        split_policy: SplitPolicy,
    ) -> None:
        coverage_rule = split_policy.coverage_rule_by_method.get(method_id)
        missing_task_ids = [
            task_id
            for task_id in selected_task_ids
            if task_id not in asset_manifest.asset_coverage_by_task
        ]
        if missing_task_ids and coverage_rule is not None and coverage_rule.allowed_to_truncate:
            return
        if missing_task_ids:
            raise AssertionError("selected task set exceeds declared asset coverage")


class ManifestSplitAlignmentGuard:  # implement SplitAlignmentGuard
    def assert_stage_task_sets_aligned(self, stage_graph: StageGraph, split_policy: SplitPolicy) -> None:
        for stage in stage_graph.stages:
            stage_digest = stage.boundary.config_fields.get("task_set_digest")
            policy_digest = stage.boundary.config_fields.get("split_policy_task_set_digest")
            if stage_digest is not None and policy_digest is not None and stage_digest != policy_digest:
                raise AssertionError(f"stage task set drifted from split policy: {stage.stage_id.value}")
