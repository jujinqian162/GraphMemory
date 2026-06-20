from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.inference import CheckpointGraphRetrieverLoader, GraphRetrieverInference
from graph_memory.retrieval.contracts import RetrievalMethodResult
from graph_memory.retrieval.requests import GraphRankingRequest, RankingMethodRequest
from graph_memory.retrieval.signals import SeedSignalProvider


@dataclass(frozen=True)
class TrainableGraphRetrievalMethod:
    """
    Retrieval-owned adapter for checkpoint-backed graph retriever inference.
    checkpoint-backed graph retriever inference 的 retrieval 适配器。
    """

    name: str
    inference: GraphRetrieverInference

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        text_embedding_provider: TextEmbeddingProvider,
        seed_signal_provider: SeedSignalProvider,
        device: str | torch.device = "cpu",
    ) -> "TrainableGraphRetrievalMethod":
        inference = CheckpointGraphRetrieverLoader().load(
            checkpoint_path,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            device=device,
        )
        return cls(name=inference.name, inference=inference)

    def rank_task(self, request: RankingMethodRequest, *, top_k: int) -> RetrievalMethodResult:
        if not isinstance(request, GraphRankingRequest):
            raise TypeError(f"{self.name} requires GraphRankingRequest, got {type(request).__name__}.")
        return self.inference.rank_task(request, top_k=top_k)
