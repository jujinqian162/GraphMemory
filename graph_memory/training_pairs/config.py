from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """
    Configuration for deterministic train pair negative sampling.
    确定性训练 pair 负采样配置。
    """

    random_seed: int = 13
    easy_random_per_positive: int = 2
    hard_bm25_per_positive: int = 2
    hard_dense_per_positive: int = 2
    hard_graph_neighbor_per_positive: int = 1
    hard_pool_size: int = 30


__all__ = ["NegativeSamplingConfig"]
