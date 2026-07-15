from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProvenanceConfig:
    seed_top_s: int = 10
    beam_width: int = 8
    max_hops: int = 4
    top_paths: int = 5
    max_path_expansions: int = 256
    semantic_weight: float = 0.45
    dependency_weight: float = 0.25
    binding_weight: float = 0.2
    grounding_weight: float = 0.1
    hop_penalty: float = 0.04
    invalidation_penalty: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "seed_top_s",
            "beam_width",
            "max_hops",
            "top_paths",
            "max_path_expansions",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "semantic_weight",
            "dependency_weight",
            "binding_weight",
            "grounding_weight",
            "hop_penalty",
            "invalidation_penalty",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


__all__ = ["ExecutionProvenanceConfig"]
