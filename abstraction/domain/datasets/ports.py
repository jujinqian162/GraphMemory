from __future__ import annotations

from typing import Protocol, Sequence

from abstraction.domain.common.identifiers import DatasetId
from abstraction.domain.datasets.definitions import (
    AssetManifest,
    DatasetDefinition,
    DatasetRecordSet,
    OfficialSplitMetadata,
    RawDatasetSource,
)
from abstraction.domain.datasets.split_policy import SplitPolicy
from abstraction.domain.task_views.eval_views import EvalLabelView
from abstraction.domain.task_views.views import TaskView


class DatasetAdapter(Protocol):
    def describe_dataset(self) -> DatasetDefinition:
        ...

    def load_raw_records(self, source: RawDatasetSource, policy: SplitPolicy) -> DatasetRecordSet:
        ...

    def describe_official_splits(self, source: RawDatasetSource) -> OfficialSplitMetadata:
        ...

    def describe_assets(self, source: RawDatasetSource, policy: SplitPolicy) -> AssetManifest:
        ...

    def build_task_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[TaskView]:
        ...

    def build_eval_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[EvalLabelView]:
        ...


class DatasetRegistry(Protocol):
    def register_dataset(self, adapter: DatasetAdapter) -> None:
        ...

    def get_dataset(self, dataset_id: DatasetId) -> DatasetAdapter:
        ...


class HotpotQADatasetAdapter:  # implement DatasetAdapter
    def describe_dataset(self) -> DatasetDefinition:
        pass

    def load_raw_records(self, source: RawDatasetSource, policy: SplitPolicy) -> DatasetRecordSet:
        pass

    def describe_official_splits(self, source: RawDatasetSource) -> OfficialSplitMetadata:
        pass

    def describe_assets(self, source: RawDatasetSource, policy: SplitPolicy) -> AssetManifest:
        pass

    def build_task_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[TaskView]:
        pass

    def build_eval_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[EvalLabelView]:
        pass


class LongMemEvalDatasetAdapter:  # implement DatasetAdapter
    def describe_dataset(self) -> DatasetDefinition:
        pass

    def load_raw_records(self, source: RawDatasetSource, policy: SplitPolicy) -> DatasetRecordSet:
        pass

    def describe_official_splits(self, source: RawDatasetSource) -> OfficialSplitMetadata:
        pass

    def describe_assets(self, source: RawDatasetSource, policy: SplitPolicy) -> AssetManifest:
        pass

    def build_task_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[TaskView]:
        pass

    def build_eval_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[EvalLabelView]:
        pass


class TwoWikiDatasetAdapter:  # implement DatasetAdapter
    def describe_dataset(self) -> DatasetDefinition:
        pass

    def load_raw_records(self, source: RawDatasetSource, policy: SplitPolicy) -> DatasetRecordSet:
        pass

    def describe_official_splits(self, source: RawDatasetSource) -> OfficialSplitMetadata:
        pass

    def describe_assets(self, source: RawDatasetSource, policy: SplitPolicy) -> AssetManifest:
        pass

    def build_task_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[TaskView]:
        pass

    def build_eval_views(self, records: DatasetRecordSet, policy: SplitPolicy) -> Sequence[EvalLabelView]:
        pass


class InMemoryDatasetRegistry:  # implement DatasetRegistry
    def __init__(self) -> None:
        self.adapters_by_dataset_id: dict[DatasetId, DatasetAdapter] = {}

    def register_dataset(self, adapter: DatasetAdapter) -> None:
        dataset_definition = adapter.describe_dataset()
        self.adapters_by_dataset_id[dataset_definition.dataset_id] = adapter

    def get_dataset(self, dataset_id: DatasetId) -> DatasetAdapter:
        return self.adapters_by_dataset_id[dataset_id]
