from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from abstraction.domain.common.capability_names import StageKind
from abstraction.domain.common.identifiers import StageId


@dataclass(frozen=True)
class StageConfigBoundary:
    stage_id: StageId
    stage_kind: StageKind
    reads_labels: bool
    reads_retrieval_visible_inputs: bool
    writes_prediction: bool
    read_artifacts: Sequence[str]
    write_artifacts: Sequence[str]
    config_fields: Mapping[str, str]


@dataclass(frozen=True)
class StagePlan:
    stage_id: StageId
    stage_kind: StageKind
    depends_on: Sequence[StageId]
    boundary: StageConfigBoundary


@dataclass(frozen=True)
class StageGraph:
    stages: Sequence[StagePlan]
