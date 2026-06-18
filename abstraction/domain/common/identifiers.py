from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetId:
    value: str


@dataclass(frozen=True)
class TaskId:
    value: str


@dataclass(frozen=True)
class ItemId:
    value: str


@dataclass(frozen=True)
class ViewId:
    value: str


@dataclass(frozen=True)
class RequestId:
    value: str


@dataclass(frozen=True)
class PredictionId:
    value: str


@dataclass(frozen=True)
class MethodId:
    value: str


@dataclass(frozen=True)
class MetricSuiteId:
    value: str


@dataclass(frozen=True)
class SplitName:
    value: str


@dataclass(frozen=True)
class StageId:
    value: str


@dataclass(frozen=True)
class ArtifactId:
    value: str

