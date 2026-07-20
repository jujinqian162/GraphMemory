from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Literal

from graph_memory.contracts.common import JsonValue
from graph_memory.retrieval.contracts import RankedNode, SeedRanker
from graph_memory.retrieval.methods.flat.bm25 import BM25TaskRetriever
from graph_memory.retrieval.requests import TextRankingRequest

ProvenanceSemanticStrategy = Literal["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class ProvenanceGraphConstructionConfig:
    strategy: ProvenanceSemanticStrategy = "bm25"
    successors_per_output: int = 2
    hybrid_dense_weight: float = 0.5
    scorer_identity: str = "provenance_semantic_v3"
    query_template_version: str = "question_source_v1"
    semantic_temperature: float = 0.1
    weight_floor: float = 0.5
    branch_policy_version: str = "rank_banded_v1"
    near_rank_bucket: tuple[int, int] = (2, 4)
    mid_rank_bucket: tuple[int, int] = (5, 8)
    tail_rank_bucket: tuple[int, int | None] = (9, None)

    def __post_init__(self) -> None:
        if self.strategy not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported provenance graph strategy={self.strategy!r}.")
        if self.successors_per_output != 2:
            raise ValueError("schema-v3 successors_per_output must be exactly 2.")
        if not 0.0 <= self.hybrid_dense_weight <= 1.0:
            raise ValueError("hybrid_dense_weight must be in [0, 1].")
        if not self.scorer_identity:
            raise ValueError("scorer_identity must be non-empty.")
        if self.query_template_version != "question_source_v1":
            raise ValueError(
                "Unsupported provenance query_template_version="
                f"{self.query_template_version!r}."
            )
        if self.semantic_temperature <= 0.0:
            raise ValueError("semantic_temperature must be positive.")
        if not 0.0 <= self.weight_floor < 1.0:
            raise ValueError("weight_floor must be in [0, 1).")
        if not self.branch_policy_version:
            raise ValueError("branch_policy_version must be non-empty.")
        previous_end = 1
        for name, (start, end) in self.rank_buckets:
            if start < 2 or (end is not None and end < start):
                raise ValueError(f"Invalid {name} rank bucket={(start, end)!r}.")
            if start <= previous_end:
                raise ValueError("Rank buckets must be ordered and non-overlapping.")
            previous_end = end if end is not None else start

    @property
    def rank_buckets(self) -> tuple[tuple[str, tuple[int, int | None]], ...]:
        return (
            ("near", self.near_rank_bucket),
            ("mid", self.mid_rank_bucket),
            ("tail", self.tail_rank_bucket),
        )

    def identity(self) -> dict[str, JsonValue]:
        return {
            "strategy": self.strategy,
            "successors_per_output": self.successors_per_output,
            "hybrid_dense_weight": self.hybrid_dense_weight,
            "scorer_identity": self.scorer_identity,
            "query_template_version": self.query_template_version,
            "semantic_temperature": self.semantic_temperature,
            "weight_floor": self.weight_floor,
            "branch_policy_version": self.branch_policy_version,
            "rank_buckets": {
                name: [bounds[0], bounds[1]] for name, bounds in self.rank_buckets
            },
        }


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
        return self.rank_many([request])[0]

    def rank_many(
        self, requests: Sequence[TextRankingRequest]
    ) -> list[list[RankedNode]]:
        if self.config.strategy == "bm25":
            return [
                _validate_complete_ranking(request, ranked)
                for request, ranked in zip(
                    requests,
                    _rank_many(self.bm25_ranker, requests),
                    strict=True,
                )
            ]
        assert self.dense_ranker is not None
        dense_many = [
            _validate_complete_ranking(request, ranked)
            for request, ranked in zip(
                requests,
                _rank_many(self.dense_ranker, requests),
                strict=True,
            )
        ]
        if self.config.strategy == "dense":
            return dense_many
        bm25_many = [
            _validate_complete_ranking(request, ranked)
            for request, ranked in zip(
                requests,
                _rank_many(self.bm25_ranker, requests),
                strict=True,
            )
        ]
        return [
            _hybrid_ranking(
                request,
                bm25=bm25,
                dense=dense,
                dense_weight=self.config.hybrid_dense_weight,
            )
            for request, bm25, dense in zip(
                requests, bm25_many, dense_many, strict=True
            )
        ]


def _rank_many(
    ranker: SeedRanker,
    requests: Sequence[TextRankingRequest],
) -> list[list[RankedNode]]:
    batch_rank = getattr(ranker, "rank_many", None)
    if callable(batch_rank):
        ranked = batch_rank(list(requests))
        if not isinstance(ranked, list) or len(ranked) != len(requests):
            raise ValueError("Provenance batch ranker returned the wrong request count.")
        return ranked
    return [ranker.rank(request) for request in requests]


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
