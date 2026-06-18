from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from abstraction.domain.common.artifacts import ArtifactRef
from abstraction.domain.common.identifiers import ItemId, MethodId


@dataclass(frozen=True)
class TemporalMemorySignals:
    method_id: MethodId
    signal_artifacts: Sequence[ArtifactRef]
    recency_signal_by_item: Mapping[ItemId, float]
    importance_signal_by_item: Mapping[ItemId, float]
