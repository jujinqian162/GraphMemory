from __future__ import annotations

from graph_memory.contracts.model import DomainModel, NonNegativeInt, PositiveInt


class NegativeSamplingConfig(DomainModel):
    """Deterministic train-pair negative sampling configuration."""

    random_seed: int = 13
    easy_random_per_positive: NonNegativeInt = 2
    hard_bm25_per_positive: NonNegativeInt = 2
    hard_dense_per_positive: NonNegativeInt = 2
    hard_graph_neighbor_per_positive: NonNegativeInt = 1
    hard_pool_size: PositiveInt = 30


class ProvenanceNegativeSamplingConfig(NegativeSamplingConfig):
    easy_random_per_positive: NonNegativeInt = 2
    hard_bm25_per_positive: NonNegativeInt = 1
    hard_dense_per_positive: NonNegativeInt = 1
    hard_graph_neighbor_per_positive: NonNegativeInt = 0
    hard_provenance_successor_per_positive: NonNegativeInt = 2
    hard_provenance_predecessor_per_positive: NonNegativeInt = 1


__all__ = ["NegativeSamplingConfig", "ProvenanceNegativeSamplingConfig"]
