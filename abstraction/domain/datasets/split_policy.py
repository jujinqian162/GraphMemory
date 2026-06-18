from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from abstraction.domain.common.capability_names import SplitRole
from abstraction.domain.common.identifiers import DatasetId, MethodId, SplitName, TaskId


@dataclass(frozen=True)
class CoverageRule:
    owner_method_id: MethodId | None
    allowed_to_truncate: bool
    required_asset_groups: Sequence[str]
    reason: str


@dataclass(frozen=True)
class BenchmarkRecipe:
    dataset_id: DatasetId
    recipe_name: str
    split_source: str
    split_count: int | None
    split_seed: int | None
    split_offset: int | None
    asset_subset: Sequence[str]
    coverage_rules: Sequence[CoverageRule]


@dataclass(frozen=True)
class SplitPolicy:
    dataset_id: DatasetId
    split_name: SplitName
    split_role: SplitRole
    selected_task_ids: Sequence[TaskId]
    label_visible_to_contexts: Sequence[str]
    coverage_rule_by_method: Mapping[MethodId, CoverageRule]


class SplitPolicyResolver(Protocol):
    def resolve_split_policy(self, recipe: BenchmarkRecipe, method_id: MethodId) -> SplitPolicy:
        ...


class ManifestSplitPolicyResolver:  # implement SplitPolicyResolver
    def resolve_split_policy(self, recipe: BenchmarkRecipe, method_id: MethodId) -> SplitPolicy:
        pass

