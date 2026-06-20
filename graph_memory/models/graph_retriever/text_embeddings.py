from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor

from graph_memory.embeddings import (
    DenseEncodingService,
    DenseTaskEncodingRequest,
    DenseTaskEncodingResult,
    SentenceEncoder,
    load_sentence_transformer,
)
from graph_memory.models.graph_retriever.contracts import TaskGraphFeatures
from graph_memory.retrieval.contracts import RankedNode
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignal, seed_signals_from_ranked_nodes


@dataclass(frozen=True)
class DenseGraphFeatureProvider:
    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    batch_size: int = 64
    encoder: SentenceEncoder | None = None
    embedding_dim: int = field(init=False, default=0)
    encoding_service: DenseEncodingService = field(init=False, repr=False)

    def __post_init__(self) -> None:
        encoder = self.encoder if self.encoder is not None else self._load_encoder(self.model_name)
        object.__setattr__(self, "encoder", encoder)
        service = DenseEncodingService(
            encoder=encoder,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
            batch_size=self.batch_size,
        )
        object.__setattr__(self, "encoding_service", service)
        object.__setattr__(self, "embedding_dim", service.embedding_dim)

    def encode_task_nodes(self, request: TextRankingRequest, node_ids: list[str]) -> Tensor:
        return self.encode_task_node_groups(
            [DenseTaskEncodingRequest(ranking_request=request, node_ids=tuple(node_ids))]
        )[0]

    def encode_task_node_groups(
        self,
        requests: Sequence[DenseTaskEncodingRequest],
    ) -> list[Tensor]:
        return [
            self._tensor_from_result(result)
            for result in self.encoding_service.encode_tasks(requests)
        ]

    def score_task(self, request: TextRankingRequest) -> list[SeedSignal]:
        return self.score_tasks([request])[0]

    def score_tasks(self, requests: Sequence[TextRankingRequest]) -> list[list[SeedSignal]]:
        encoding_requests = [
            DenseTaskEncodingRequest(
                ranking_request=request,
                node_ids=("q", *(candidate.item_id for candidate in request.candidates)),
            )
            for request in requests
        ]
        return [
            features.seed_signals
            for features in self.build_task_feature_groups(encoding_requests)
        ]

    def build_task_feature_groups(
        self,
        requests: Sequence[DenseTaskEncodingRequest],
    ) -> list[TaskGraphFeatures]:
        return [
            TaskGraphFeatures(
                node_embeddings=self._tensor_from_result(result),
                seed_signals=self._seed_signals(request.ranking_request, result),
            )
            for request, result in zip(
                requests,
                self.encoding_service.encode_tasks(requests),
                strict=True,
            )
        ]

    @staticmethod
    def _tensor_from_result(result: DenseTaskEncodingResult) -> Tensor:
        return torch.tensor(result.embeddings, dtype=torch.float32)

    @staticmethod
    def _seed_signals(
        request: TextRankingRequest,
        result: DenseTaskEncodingResult,
    ) -> list[SeedSignal]:
        row_by_node_id = {
            node_id: result.embeddings[index]
            for index, node_id in enumerate(result.node_ids)
        }
        if "q" not in row_by_node_id:
            raise ValueError(f"Dense graph features require q node for task_id={request.task_id}.")
        try:
            ranked_nodes = [
                RankedNode(
                    node_id=candidate.item_id,
                    score=float(row_by_node_id[candidate.item_id] @ row_by_node_id["q"]),
                )
                for candidate in request.candidates
            ]
        except KeyError as error:
            raise ValueError(
                f"Dense graph features are missing node_id={error.args[0]} "
                f"for task_id={request.task_id}."
            ) from error
        return seed_signals_from_ranked_nodes(request, ranked_nodes)

    @staticmethod
    def _load_encoder(model_name: str) -> SentenceEncoder:
        try:
            return cast(SentenceEncoder, cast(object, load_sentence_transformer(model_name)))
        except RuntimeError as error:
            raise RuntimeError(
                "sentence-transformers is required for trainable retrieval unless an embedding provider is injected."
            ) from error


__all__ = ["DenseGraphFeatureProvider"]
