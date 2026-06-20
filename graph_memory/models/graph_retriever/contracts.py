from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from graph_memory.embeddings import DenseTaskEncodingRequest
from graph_memory.retrieval.requests import TextRankingRequest
from graph_memory.retrieval.signals import SeedSignal, SeedSignalProvider, score_tasks


class TextEmbeddingProvider(Protocol):
    """
    Replaceable provider for frozen query and candidate text embeddings.
    可替换的冻结 query 和 candidate 文本 embedding 提供器。
    """

    @property
    def embedding_dim(self) -> int:
        ...

    def encode_task_nodes(self, request: TextRankingRequest, node_ids: list[str]) -> Tensor:
        ...


@runtime_checkable
class BulkTextEmbeddingProvider(Protocol):
    def encode_task_node_groups(
        self,
        requests: Sequence[DenseTaskEncodingRequest],
    ) -> list[Tensor]:
        ...


@dataclass(frozen=True)
class TaskGraphFeatures:
    node_embeddings: Tensor
    seed_signals: list[SeedSignal]


@runtime_checkable
class JointGraphFeatureProvider(Protocol):
    def build_task_feature_groups(
        self,
        requests: Sequence[DenseTaskEncodingRequest],
    ) -> list[TaskGraphFeatures]:
        ...


def encode_task_node_groups(
    provider: TextEmbeddingProvider,
    requests: Sequence[DenseTaskEncodingRequest],
) -> list[Tensor]:
    request_list = list(requests)
    if isinstance(provider, BulkTextEmbeddingProvider):
        results = provider.encode_task_node_groups(request_list)
        if len(results) != len(request_list):
            raise ValueError(
                "Bulk text embedding provider returned an invalid result count: "
                f"expected={len(request_list)} observed={len(results)}."
            )
        return results
    return [
        provider.encode_task_nodes(request.ranking_request, list(request.node_ids))
        for request in request_list
    ]


def build_task_feature_groups(
    text_embedding_provider: TextEmbeddingProvider,
    seed_signal_provider: SeedSignalProvider,
    requests: Sequence[DenseTaskEncodingRequest],
) -> list[TaskGraphFeatures]:
    request_list = list(requests)
    if (
        text_embedding_provider is seed_signal_provider
        and isinstance(text_embedding_provider, JointGraphFeatureProvider)
    ):
        results = text_embedding_provider.build_task_feature_groups(request_list)
        if len(results) != len(request_list):
            raise ValueError(
                "Joint graph feature provider returned an invalid result count: "
                f"expected={len(request_list)} observed={len(results)}."
            )
        return results

    embeddings_by_task = encode_task_node_groups(text_embedding_provider, request_list)
    signals_by_task = score_tasks(
        seed_signal_provider,
        [request.ranking_request for request in request_list],
    )
    return [
        TaskGraphFeatures(node_embeddings=embeddings, seed_signals=signals)
        for embeddings, signals in zip(embeddings_by_task, signals_by_task, strict=True)
    ]


__all__ = [
    "BulkTextEmbeddingProvider",
    "JointGraphFeatureProvider",
    "TaskGraphFeatures",
    "TextEmbeddingProvider",
    "build_task_feature_groups",
    "encode_task_node_groups",
]
