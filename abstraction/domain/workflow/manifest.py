from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from abstraction.domain.common.artifacts import ArtifactRef
from abstraction.domain.common.identifiers import DatasetId, MethodId, MetricSuiteId, StageId
from abstraction.domain.datasets.definitions import RawDatasetSource
from abstraction.domain.datasets.split_policy import BenchmarkRecipe, SplitPolicy


@dataclass(frozen=True)
class ExperimentRunIntent:
    dataset_id: DatasetId
    method_id: MethodId
    metric_suite_id: MetricSuiteId
    benchmark_recipe: BenchmarkRecipe
    raw_source: RawDatasetSource


@dataclass(frozen=True)
class StageArtifact:
    stage_id: StageId
    produced_artifacts: Sequence[ArtifactRef]
    consumed_artifacts: Sequence[ArtifactRef]


@dataclass(frozen=True)
class ArtifactManifest:
    run_intent: ExperimentRunIntent
    split_policy: SplitPolicy
    stage_artifacts: Sequence[StageArtifact]
    task_set_digest_by_stage: Mapping[StageId, str]
