from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from abstraction.domain.common.identifiers import DatasetId, SplitName, TaskId


@dataclass(frozen=True)
class RawDatasetSource:
    dataset_id: DatasetId
    source_path: Path
    source_format: str
    source_version: str


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: DatasetId
    display_name: str
    supported_raw_sources: Sequence[str]
    provided_task_views: Sequence[str]
    provided_eval_views: Sequence[str]


@dataclass(frozen=True)
class DatasetRecordSet:
    dataset_id: DatasetId
    split_name: SplitName
    task_ids: Sequence[TaskId]
    record_count: int
    record_manifest_path: Path


@dataclass(frozen=True)
class AssetManifest:
    dataset_id: DatasetId
    asset_root: Path
    available_asset_groups: Sequence[str]
    asset_coverage_by_task: Mapping[TaskId, Sequence[str]]


@dataclass(frozen=True)
class OfficialSplitMetadata:
    dataset_id: DatasetId
    split_names: Sequence[SplitName]
    split_source: str
    stable_id_field: str

