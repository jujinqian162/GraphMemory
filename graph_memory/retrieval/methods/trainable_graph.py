from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from graph_memory.contracts.graphs import GraphEdge, MemoryGraph
from graph_memory.contracts.tasks import MemoryTaskInput
from graph_memory.models.graph_retriever.contracts import TextEmbeddingProvider
from graph_memory.models.graph_retriever.inference import CheckpointGraphRetrieverLoader, GraphRetrieverInference
from graph_memory.retrieval.contracts import RankedNode
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
        graphs: list[MemoryGraph],
        text_embedding_provider: TextEmbeddingProvider | None = None,
        seed_signal_provider: SeedSignalProvider | None = None,
        dense_encoder: object | None = None,
        device: str | torch.device = "cpu",
    ) -> "TrainableGraphRetrievalMethod":
        inference = CheckpointGraphRetrieverLoader().load(
            checkpoint_path,
            graphs=graphs,
            text_embedding_provider=text_embedding_provider,
            seed_signal_provider=seed_signal_provider,
            dense_encoder=dense_encoder,
            device=device,
        )
        return cls(name=inference.name, inference=inference)

    def rank_task(self, task_input: MemoryTaskInput, *, top_k: int) -> tuple[list[RankedNode], list[GraphEdge]]:
        return self.inference.rank_task(task_input, top_k=top_k)
