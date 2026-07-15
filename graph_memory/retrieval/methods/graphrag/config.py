from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphRAGConfig:
    seed_top_s: int = 8
    restart_probability: float = 0.2
    max_iterations: int = 30
    convergence_tolerance: float = 1e-6
    semantic_weight: float = 0.45
    entity_weight: float = 0.55
    min_entity_length: int = 2
    max_entity_words: int = 8

    def __post_init__(self) -> None:
        if self.seed_top_s <= 0:
            raise ValueError("seed_top_s must be positive.")
        if not 0.0 < self.restart_probability <= 1.0:
            raise ValueError("restart_probability must be in (0, 1].")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive.")
        if self.semantic_weight < 0.0 or self.entity_weight < 0.0:
            raise ValueError("GraphRAG score weights must be non-negative.")
        if self.semantic_weight + self.entity_weight == 0.0:
            raise ValueError("At least one GraphRAG score weight must be positive.")
        if self.min_entity_length <= 0:
            raise ValueError("min_entity_length must be positive.")
        if self.max_entity_words <= 0:
            raise ValueError("max_entity_words must be positive.")


__all__ = ["GraphRAGConfig"]
