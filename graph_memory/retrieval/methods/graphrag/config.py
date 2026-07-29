from __future__ import annotations

from typing import Literal

from pydantic import Field

from graph_memory.contracts.model import (
    DomainModel,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveInt,
)


class GraphRAGConfig(DomainModel):
    seed_top_s: PositiveInt = 5
    max_entity_document_frequency_ratio: float = Field(
        default=0.25, allow_inf_nan=False, gt=0.0, le=1.0
    )
    sentence_resolver: Literal["frozen_dense"] = "frozen_dense"
    min_sentence_score_margin: NonNegativeFiniteFloat = 0.02
    min_bridge_confidence: NonNegativeFiniteFloat = 0.2
    max_partners_per_anchor: Literal[1] = 1
    preserve_dense_top_n: NonNegativeInt = 2


__all__ = ["GraphRAGConfig"]
