from __future__ import annotations

from graph_memory.contracts.model import (
    DomainModel,
    FiniteFloat,
    NonEmptyStr,
)
from graph_memory.retrieval.methods.ids import DenseCandidateView


class CrossEncoderExample(DomainModel):
    task_id: NonEmptyStr
    candidate_id: NonEmptyStr
    query_text: str
    candidate_text: str
    label: FiniteFloat


class CrossEncoderTrainingResult(DomainModel):
    model_dir: str
    selected_metric_name: NonEmptyStr
    selected_metric_value: FiniteFloat
    metric_records: tuple[dict[str, object], ...]
    variant: DenseCandidateView


__all__ = ["CrossEncoderExample", "CrossEncoderTrainingResult"]
