from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProvenanceConfig:
    seed_top_s: int = 5
    beam_width: int = 8
    max_hops: int = 2
    max_paths_per_seed: int = 1
    max_path_expansions: int = 256
    min_path_confidence: float = 0.2
    preserve_dense_top_n: int = 2
    hop_penalty: float = 0.04

    def __post_init__(self) -> None:
        for name in (
            "seed_top_s",
            "beam_width",
            "max_hops",
            "max_paths_per_seed",
            "max_path_expansions",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_paths_per_seed != 1:
            raise ValueError("max_paths_per_seed must be exactly 1.")
        if self.preserve_dense_top_n < 0:
            raise ValueError("preserve_dense_top_n must be non-negative.")
        for name in ("min_path_confidence", "hop_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


__all__ = ["ExecutionProvenanceConfig"]
