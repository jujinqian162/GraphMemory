from __future__ import annotations

from pydantic import Field, StrictBool, model_validator

from graph_memory.contracts.model import (
    DomainModel,
    NonNegativeFiniteFloat,
    NonNegativeInt,
    PositiveFiniteFloat,
    PositiveInt,
)


class GraphRAGConfig(DomainModel):
    """Deterministic retrieval configuration for the FastGraphRAG-style baseline."""

    text_unit_size: PositiveInt = 100
    text_unit_overlap: NonNegativeInt = 20
    max_entity_words: PositiveInt = 4
    max_word_length: PositiveInt = 15
    max_entities_per_text_unit: PositiveInt = 48
    min_node_frequency: PositiveInt = 2
    min_node_degree: PositiveInt = 1
    min_edge_weight_percentile: float = Field(
        default=40.0, allow_inf_nan=False, ge=0.0, le=100.0
    )
    remove_ego_node: StrictBool = True
    normalize_edge_weights: StrictBool = True

    seed_top_s: PositiveInt = 10
    exact_match_weight: NonNegativeFiniteFloat = 1.0
    lexical_weight: NonNegativeFiniteFloat = 0.5
    semantic_seed_weight: NonNegativeFiniteFloat = 1.0
    restart_probability: float = Field(default=0.2, allow_inf_nan=False, gt=0.0, le=1.0)
    max_iterations: PositiveInt = 30
    convergence_tolerance: PositiveFiniteFloat = 1e-6

    semantic_weight: NonNegativeFiniteFloat = 0.5
    graph_weight: NonNegativeFiniteFloat = 0.5
    trace_top_entities: PositiveInt = 50

    @model_validator(mode="after")
    def _validate_config(self) -> "GraphRAGConfig":
        if self.text_unit_overlap >= self.text_unit_size:
            raise ValueError(
                "GraphRAG text_unit_overlap must be smaller than text_unit_size"
            )
        if self.semantic_weight + self.graph_weight <= 0.0:
            raise ValueError("GraphRAG score weights cannot both be zero")
        if (
            self.exact_match_weight + self.lexical_weight + self.semantic_seed_weight
            <= 0.0
        ):
            raise ValueError("GraphRAG seed weights cannot all be zero")
        return self


__all__ = ["GraphRAGConfig"]
