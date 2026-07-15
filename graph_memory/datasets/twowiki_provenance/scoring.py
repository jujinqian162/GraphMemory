from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from graph_memory.retrieval.contracts import RankedNode, SeedRanker
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.requests import TextRankingRequest

ProvenanceSemanticStrategy = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class ProvenanceGraphConstructionConfig:
    strategy: ProvenanceSemanticStrategy = "bm25"
    successors_per_output: int = 2
    hybrid_dense_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.strategy not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported provenance graph strategy={self.strategy!r}.")
        if self.successors_per_output <= 0:
            raise ValueError("successors_per_output must be positive.")
        if not 0.0 <= self.hybrid_dense_weight <= 1.0:
            raise ValueError("hybrid_dense_weight must be in [0, 1].")


@dataclass(frozen=True)
class ProvenanceSemanticRanker:
    config: ProvenanceGraphConstructionConfig
    dense_ranker: SeedRanker | None = None
    bm25_ranker: SeedRanker = field(default_factory=BM25TaskRetriever)

    def __post_init__(self) -> None:
        if self.config.strategy in {"dense", "hybrid"} and self.dense_ranker is None:
            raise ValueError(
                f"Provenance graph strategy={self.config.strategy!r} requires a dense ranker."
            )

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        if self.config.strategy == "bm25":
            return _validate_complete_ranking(request, self.bm25_ranker.rank(request))
        assert self.dense_ranker is not None
        dense = _validate_complete_ranking(request, self.dense_ranker.rank(request))
        if self.config.strategy == "dense":
            return dense
        bm25 = _validate_complete_ranking(request, self.bm25_ranker.rank(request))
        return _hybrid_ranking(
            request,
            bm25=bm25,
            dense=dense,
            dense_weight=self.config.hybrid_dense_weight,
        )


def _hybrid_ranking(
    request: TextRankingRequest,
    *,
    bm25: list[RankedNode],
    dense: list[RankedNode],
    dense_weight: float,
) -> list[RankedNode]:
    bm25_score = _rank_percentiles(bm25)
    dense_score = _rank_percentiles(dense)
    combined = [
        RankedNode(
            candidate.item_id,
            (1.0 - dense_weight) * bm25_score[candidate.item_id]
            + dense_weight * dense_score[candidate.item_id],
        )
        for candidate in request.candidates
    ]
    return sorted(combined, key=lambda item: (-item.score, item.node_id))


def _rank_percentiles(ranked: list[RankedNode]) -> dict[str, float]:
    denominator = max(1, len(ranked) - 1)
    return {
        item.node_id: 1.0 - index / denominator
        for index, item in enumerate(ranked)
    }


def _validate_complete_ranking(
    request: TextRankingRequest,
    ranked: list[RankedNode],
) -> list[RankedNode]:
    expected = {candidate.item_id for candidate in request.candidates}
    observed = {item.node_id for item in ranked}
    if len(ranked) != len(observed) or observed != expected:
        raise ValueError(
            "Provenance semantic ranker must return every candidate exactly once: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}."
        )
    return sorted(ranked, key=lambda item: (-item.score, item.node_id))


__all__ = [
    "ProvenanceGraphConstructionConfig",
    "ProvenanceSemanticRanker",
    "ProvenanceSemanticStrategy",
]
