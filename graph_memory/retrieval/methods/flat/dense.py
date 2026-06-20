from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from graph_memory.embeddings import (
    DenseEncodingService,
    DenseTaskEncodingRequest,
    SentenceEncoder,
    load_sentence_transformer,
)
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import DenseConfigLike, TextRankingRequest


@dataclass(frozen=True)
class DenseConfig:
    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64


class DenseTaskRetriever:
    method_name = "dense"

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 64,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        config: DenseConfigLike | None = None,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self.config = config or DenseConfig(
            model_name=model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
        )
        self.encoder = encoder if encoder is not None else self._load_encoder(self.config.model_name)
        self.encoding_service = DenseEncodingService(
            encoder=self.encoder,
            query_prefix=self.config.query_prefix,
            passage_prefix=self.config.passage_prefix,
            batch_size=self.config.batch_size,
        )

    def rank(self, request: TextRankingRequest) -> list[RankedNode]:
        return self.rank_many([request])[0]

    def rank_many(self, requests: list[TextRankingRequest]) -> list[list[RankedNode]]:
        encoding_requests = [
            DenseTaskEncodingRequest(
                ranking_request=request,
                node_ids=("q", *(candidate.item_id for candidate in request.candidates)),
            )
            for request in requests
        ]
        return [
            self._rank_from_embeddings(request.ranking_request, result.embeddings)
            for request, result in zip(encoding_requests, self.encoding_service.encode_tasks(encoding_requests), strict=True)
        ]

    @staticmethod
    def _rank_from_embeddings(
        request: TextRankingRequest,
        embeddings: np.ndarray,
    ) -> list[RankedNode]:
        query_vector = embeddings[0]
        passage_matrix = embeddings[1:]
        scores = passage_matrix @ query_vector
        ranked_nodes = [
            RankedNode(node_id=candidate.item_id, score=float(score))
            for candidate, score in zip(request.candidates, scores, strict=True)
        ]
        return sorted(ranked_nodes, key=lambda ranked_node: (-ranked_node.score, ranked_node.node_id))

    @staticmethod
    def _load_encoder(model_name: str) -> SentenceEncoder:
        try:
            return cast(SentenceEncoder, cast(object, load_sentence_transformer(model_name)))
        except RuntimeError as error:
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval unless a test encoder is provided."
            ) from error
