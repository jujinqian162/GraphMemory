from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GraphRAGConfig:
    seed_top_s: int = 5
    max_entity_document_frequency_ratio: float = 0.25
    sentence_resolver: Literal["frozen_dense"] = "frozen_dense"
    min_sentence_score_margin: float = 0.02
    min_bridge_confidence: float = 0.2
    max_partners_per_anchor: int = 1
    preserve_dense_top_n: int = 2

    def __post_init__(self) -> None:
        if self.seed_top_s <= 0:
            raise ValueError("seed_top_s must be positive.")
        if not 0.0 < self.max_entity_document_frequency_ratio <= 1.0:
            raise ValueError("max_entity_document_frequency_ratio must be in (0, 1].")
        if self.sentence_resolver != "frozen_dense":
            raise ValueError("sentence_resolver must be 'frozen_dense'.")
        for name in ("min_sentence_score_margin", "min_bridge_confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.max_partners_per_anchor != 1:
            raise ValueError("max_partners_per_anchor must be exactly 1.")
        if self.preserve_dense_top_n < 0:
            raise ValueError("preserve_dense_top_n must be non-negative.")


__all__ = ["GraphRAGConfig"]
